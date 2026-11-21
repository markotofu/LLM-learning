/* ML Atlas – content: more neural-network and LLM concepts (seq2seq, BERT, text embeddings, training at scale, reasoning, long context, MoE) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'seq2seq', branch: 'dl', title: 'Sequence-to-Sequence Models & the Origin of Attention', short: 'Seq2seq & Attention Origins', level: 3, minutes: 40, prereqs: ['rnn', 'embeddings'],
    tags: ['seq2seq', 'encoder-decoder', 'machine translation', 'teacher forcing', 'bottleneck', 'bahdanau attention', 'luong attention', 'alignment'],
    summary: 'Mapping one sequence to another: the RNN encoder–decoder used for translation, teacher forcing and step-by-step decoding, the fixed-size bottleneck problem, and how Bahdanau attention solved it by letting the decoder look back at every input word (the direct ancestor of the transformer).',
    why: `Before transformers, translation, summarization and speech recognition used **sequence-to-sequence** models: one RNN reads the input, another writes the output. Their weakness (squeezing a whole sentence into one fixed-size vector) is exactly the problem attention was invented to fix in 2014. Seeing that history makes attention feel inevitable rather than magical, and the encoder–decoder pattern lives on in T5, Whisper and every model that uses cross-attention.`,
    objectives: [
      'Describe the encoder–decoder architecture and teacher forcing',
      'Explain the information bottleneck of a single fixed-size context vector',
      'Compute additive (Bahdanau) and dot-product attention weights and the context vector for one decoder step',
      'Connect seq2seq attention to the queries, keys and values of transformers'
    ],
    sections: [
      { type: 'text', title: 'Encoder and decoder', md: `A **sequence-to-sequence** (seq2seq) model has two parts:

\`\`\`text
encoder:  the  black  cat  sat   ->  h1 h2 h3 h4   (one hidden state per input word)
          context c = h4         (the final state summarizes the whole sentence)
decoder:  <bos> -> le, le -> chat, chat -> noir, noir -> <eos>
          (each step gets the previous output word and its own hidden state, started from c)
\`\`\`

- **Training** uses **teacher forcing**: at every step the decoder is fed the *true* previous word, not its own guess, and the loss is the cross-entropy of the next word. It is fast and stable, but the model never practises recovering from its own mistakes.
- **Inference** decodes step by step (greedily or with beam search) until it produces an end-of-sequence token.

Input and output lengths can differ freely, which is what makes the design fit translation, summarization, question answering and speech recognition.` },
      { type: 'text', title: 'The bottleneck', md: `The decoder sees the input only through **one fixed-size vector** c, whether the sentence has 5 words or 50. Long inputs must be crammed into the same space, early words fade from the final hidden state, and translation quality drops sharply with sentence length. Tricks such as reversing the input order and using LSTMs helped, but did not remove the bottleneck.

The fix: instead of one summary, keep **all** encoder states and let the decoder decide, at every output step, which input words to look at.` },
      { type: 'text', title: 'Attention: look back at every input word', md: `Bahdanau et al. (2014) gave the decoder a fresh, step-specific summary. At output step t, every encoder state h_j is scored against the decoder's previous state, the scores become weights with a softmax, and the weighted sum of encoder states is the **context** for this step:

$$ e_tj = vᵀ tanh( W_s s_{t−1} + W_h h_j ) $$
$$ α_tj = exp(e_tj) / Σ_k exp(e_tk)          c_t = Σ_j α_tj h_j $$
:::where
s_{t−1} — the decoder's previous hidden state: what it is currently "looking for"
h_j — the encoder hidden state for input word j
W_s, W_h, v — learned parameters of a tiny scoring network ("additive" attention)
e_tj — the alignment score between output step t and input word j
α_tj — the attention weight: a probability distribution over the input words
c_t — the context vector for step t, a weighted average of the encoder states
:::

Luong et al. (2015) simplified the score to a **dot product**, e_tj = s · h_j, exactly the form transformers use. Plotting the weights shows learned **alignments**: when producing "chat" the decoder attends to "cat"; for "noir" it attends to "black", even though the word order differs.` },
      { type: 'code', title: 'One decoder step of attention', code: `
import numpy as np
rng = np.random.default_rng(0)
words = ["the", "black", "cat", "sat", "<eos>"]
T, d = len(words), 4
H = rng.normal(size=(T, d))                   # encoder hidden states, one per input word
H = H / np.linalg.norm(H, axis=1, keepdims=True)    # unit length, so dot products measure direction only
s = 4 * H[2] + 0.1 * rng.normal(size=d)       # a decoder state that is "looking for" the word cat

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

alpha_dot = softmax(H @ s)                     # dot-product (Luong) attention
Ws, Wh, v = rng.normal(0, 0.5, (8, d)), rng.normal(0, 0.5, (8, d)), rng.normal(0, 0.5, 8)
alpha_add = softmax(np.tanh(Ws @ s + H @ Wh.T) @ v)    # additive (Bahdanau) attention, untrained
for w, a1, a2 in zip(words, alpha_dot, alpha_add):
    print(f"{w:>6}   dot-product weight {a1:.2f}   additive weight (random parameters) {a2:.2f}")
print("context vector:", (alpha_dot @ H).round(2), "   encoder state of 'cat':", H[2].round(2))
`, after: `The dot-product weights concentrate on "cat" because the decoder state resembles its encoder state, so the context vector is close to that word's representation. The additive scorer has random parameters here; training would teach it which words to align.` },
      { type: 'check', question: { type: 'mcq', q: 'What problem did attention solve in RNN encoder–decoders?', options: ['Slow tokenization', 'Squeezing the whole input into one fixed-size vector, which loses information for long inputs', 'Too many parameters in the decoder', 'Overfitting to the training data'], answer: 1, explain: 'Each output step gets its own weighted view of all encoder states.' } },
      { type: 'text', title: 'From seq2seq attention to transformers', md: `| Seq2seq with attention | Transformer |
| --- | --- |
| decoder state s_{t−1} | **query** |
| encoder states h_j | **keys** and **values** |
| additive or dot-product score | scaled dot product, Q Kᵀ / √d |
| attention from decoder to encoder | **cross-attention** in encoder–decoder transformers |
| RNNs carry information along the sequence | **self-attention** relates all positions directly |

"Attention Is All You Need" (2017) asked: if attention is what matters, why keep the RNNs at all? Dropping recurrence made training fully parallel and let every position reach every other in one step. T5, BART and Whisper keep the encoder–decoder layout with cross-attention; GPT keeps only the decoder.` },
      { type: 'code', title: 'A GRU encoder–decoder with attention in PyTorch', lang: 'pytorch', code: `
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
    def forward(self, src):                                   # src: (B, S) token ids
        H, h = self.rnn(self.emb(src))                        # H: (B, S, d) all states; h: (1, B, d) last
        return H, h

class AttnDecoder(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = nn.GRU(2 * d, d, batch_first=True)
        self.out = nn.Linear(2 * d, vocab)
    def forward(self, prev_tok, h, H):                        # one step: prev_tok (B,), h (1, B, d), H (B, S, d)
        scores = torch.bmm(H, h[-1].unsqueeze(2)).squeeze(2)  # (B, S) dot-product attention scores
        alpha = F.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) # (B, d) weighted sum of encoder states
        x = torch.cat([self.emb(prev_tok), context], dim=1).unsqueeze(1)
        o, h = self.rnn(x, h)
        logits = self.out(torch.cat([o.squeeze(1), context], dim=1))
        return logits, h, alpha

def seq2seq_loss(enc, dec, src, tgt):                         # tgt starts with <bos> and ends with <eos>
    H, h = enc(src)
    loss = 0.0
    for t in range(tgt.shape[1] - 1):
        logits, h, _ = dec(tgt[:, t], h, H)                   # teacher forcing: feed the TRUE previous token
        loss = loss + F.cross_entropy(logits, tgt[:, t + 1])
    return loss / (tgt.shape[1] - 1)
` },
      { type: 'check', question: { type: 'tf', q: 'With teacher forcing, the decoder is fed its own previous prediction during training.', answer: false, explain: 'It is fed the true previous token; at inference time it must use its own predictions.' } },
      { type: 'text', title: "Teacher forcing, exposure bias and decoding", md: `Teacher forcing makes training fast and stable: every decoder step is trained from the correct previous word, so all the steps can be computed in parallel and errors never pile up. But it creates a mismatch called **exposure bias**: during training the decoder only ever sees perfect prefixes, while at inference it must continue from its own, sometimes wrong, outputs, a situation it never practised. One early mistake can push it into unfamiliar territory, and the errors compound.

Remedies include **scheduled sampling** (gradually feeding the model its own predictions during training), sequence-level training objectives and, above all, better decoding:
- **greedy decoding** takes the most probable word at each step: fast but short-sighted;
- **beam search** keeps the b most probable partial outputs (b = 4 to 10 is typical) and returns the best complete one, finding higher-probability outputs at a modest cost;
- a **length penalty** stops beam search from favouring short outputs, whose few word probabilities multiply to a larger number.

Modern LLMs inherit all of this; see [Decoding: Greedy, Sampling, Temperature, Top-k & Top-p](#/concept/decoding).` },
      { type: 'text', title: "Measuring translations: BLEU", md: `How do you score a translation automatically? **BLEU** compares the candidate with one or more human reference translations by counting overlapping n-grams (single words, pairs, triples and quadruples):

$$ BLEU = BP · exp( (1/4) Σ_{n=1..4} log pₙ )          BP = min( 1, exp(1 − r / c) ) $$
:::where
pₙ — the modified n-gram precision: the share of the candidate's n-grams that also appear in the reference, each counted at most as often as it appears there (so "the the the" cannot score by repetition)
exp((1/4) Σ log pₙ) — the geometric mean of the four precisions
BP — the brevity penalty, which punishes candidates shorter than the reference
c — the length of the candidate; r — the length of the reference
:::

BLEU is cheap and agrees reasonably with human judgement when averaged over a whole test set, but it is blind to meaning: a correct paraphrase in different words scores badly, and the BLEU of a single sentence is very noisy. Newer metrics compare embeddings (BERTScore) or use trained models (COMET), and open-ended generation is usually judged by people or by LLM judges (see [Evaluating LLM Applications](#/concept/llm-evaluation)).` },
      { type: 'code', title: "BLEU from scratch", code: `import math
from collections import Counter

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def bleu(candidate, reference, max_n=4):
    c, r = candidate.split(), reference.split()
    precisions = []
    for n in range(1, max_n + 1):
        cand, ref = ngrams(c, n), ngrams(r, n)
        overlap = sum(min(count, ref[g]) for g, count in cand.items())   # clipped counts
        precisions.append(overlap / max(sum(cand.values()), 1))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    if min(precisions) == 0:
        return 0.0, precisions, bp
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n), precisions, bp

reference = "the black cat sat quietly on the old mat"
candidates = ["the black cat sat quietly on the old mat",
              "the black cat sat on the old mat",
              "the black cat sat quietly",
              "a dark cat was sitting on the mat",
              "the the the the the the the the the"]
for cand in candidates:
    score, p, bp = bleu(cand, reference)
    print(f"BLEU {score:.3f}  precisions {[round(x, 2) for x in p]}  brevity penalty {bp:.3f}  | {cand}")
score, p, bp = bleu("the black cat sat quietly", reference)
print("short candidate penalised only by brevity:", min(p) == 1.0 and bp < 1)`, after: `A perfect copy scores 1. Dropping one word costs some n-gram matches and a little brevity penalty. The five-word prefix has perfect precision at every n, yet scores only about 0.45, because the brevity penalty catches the missing words. The paraphrase, which a person would accept, scores 0 because it shares no 3-gram with the reference. And clipping stops the repeated "the" from earning credit beyond its two occurrences in the reference.`, predict: {"type":"mcq","q":"The candidate 'the black cat sat quietly' matches the first five words of the reference exactly. What keeps its BLEU score well below 1?","options":["Its n-gram precisions","The brevity penalty","Clipping","Nothing: it scores 1"],"answer":1,"explain":"Every n-gram of the candidate appears in the reference, so all precisions are 1; BLEU punishes the missing words only through the brevity penalty exp(1 − 9/5) ≈ 0.45.","expect":["short candidate penalised only by brevity: True"]} }
    ],
    when: `- Encoder–decoder models when input and output are different sequences: translation, summarization, speech-to-text (Whisper), data-to-text.
- RNN seq2seq is mostly historical now, but it is still useful on tiny devices and for teaching.
- Cross-attention is how one modality conditions another: text conditioning image diffusion, a decoder reading an encoder's output.`,
    pitfalls: `- Exposure bias: teacher forcing means the model never trains on its own mistakes; scheduled sampling or sequence-level training can help.
- Forgetting the end-of-sequence token, so decoding never stops.
- Mismatched special tokens or padding between encoder and decoder vocabularies.
- Attending to padding positions: mask them out before the softmax.`,
    takeaways: [
      'Seq2seq: an encoder reads the input, a decoder generates the output token by token.',
      'Teacher forcing feeds the true previous token during training.',
      'One fixed-size context vector is a bottleneck for long inputs.',
      'Attention gives each output step its own weighted average of all encoder states.',
      'Decoder state = query, encoder states = keys and values: the step that led to the transformer.'
    ],
    quiz: [
      { type: 'mcq', q: 'In a seq2seq model with attention, what is recomputed at every decoder step?', options: ['The encoder states', 'The attention weights and the context vector', 'The vocabulary', 'The embedding matrix'], answer: 1, explain: 'Each output word looks at the input differently.' },
      { type: 'mcq', q: 'In transformer terms, the encoder states in seq2seq attention play the role of…', options: ['queries', 'keys and values', 'positional encodings', 'the loss'], answer: 1, explain: 'The decoder state is the query.' },
      { type: 'tf', q: 'Seq2seq models can produce outputs of a different length than their inputs.', answer: true, explain: 'The decoder runs until it emits an end token.' },
      { type: 'mcq', q: 'Which modern model keeps the encoder–decoder layout with cross-attention?', options: ['GPT-2', 'Whisper', 'BERT', 'Word2vec'], answer: 1, explain: 'Whisper encodes audio and decodes text with cross-attention.' },
      { type: 'order', q: 'Order one decoder step with attention.', items: ['Score every encoder state against the decoder state', 'Softmax the scores into weights', 'Take the weighted sum of encoder states (context)', 'Combine the context with the decoder state to predict the next word'], explain: 'Score → weights → context → prediction.' },
      {"type":"mcq","q":"What is exposure bias?","options":["Bias in the training data","The decoder is trained only on correct prefixes but must continue from its own, possibly wrong, outputs at inference","Attention focusing on a single word","Overfitting to long sentences"],"answer":1,"explain":"Teacher forcing never shows the decoder its own mistakes, so it has not learned to recover from them."},
      {"type":"mcq","q":"Why does BLEU clip each n-gram's count by its count in the reference?","options":["To make it faster","So that a candidate cannot score by repeating a correct word many times","To reward long sentences","To ignore punctuation"],"answer":1,"explain":"Without clipping, 'the the the the' would get perfect unigram precision against any reference containing 'the'."}
    ],
    problems: [
      {
        id: 'additive-attention', title: 'Additive (Bahdanau) Attention', difficulty: 'medium',
        prompt: `Implement \`additive_attention(s, H, Ws, Wh, v)\` → \`(weights, context)\` for one decoder step:

$$ e_j = v · tanh( Ws s + Wh h_j )          α = softmax(e)          c = Σ_j α_j h_j $$
:::where
s — the decoder state, shape (d_s,)
H — the encoder states, one row h_j per input word, shape (T, d_h)
Ws, Wh — learned matrices of shapes (d_a, d_s) and (d_a, d_h)
v — a learned vector of length d_a
α — the attention weights (length T, summing to 1); c — the context vector (length d_h)
:::

Compute all T scores at once (\`H @ Wh.T\` has shape (T, d_a)) and use a numerically stable softmax.

\`\`\`text
With v = 0 every score is 0, so the weights are uniform and c is the mean of the rows of H.
\`\`\``,
        starter: `
import numpy as np

def additive_attention(s, H, Ws, Wh, v):
    # TODO
    pass
`,
        solution: `
import numpy as np

def additive_attention(s, H, Ws, Wh, v):
    s = np.asarray(s, dtype=float)
    H = np.asarray(H, dtype=float)
    e = np.tanh(Ws @ s + H @ Wh.T) @ v
    e = e - e.max()
    alpha = np.exp(e) / np.exp(e).sum()
    return alpha, alpha @ H
`,
        hints: ['`Ws @ s` has shape (d_a,) and broadcasts against `H @ Wh.T` of shape (T, d_a).', 'Subtract the maximum score before `np.exp`.', 'The context is `alpha @ H`.'],
        explain: `Vectorizing over all T encoder states makes one decoder step a couple of matrix products. Swapping the tanh scorer for a plain dot product gives Luong attention, and, with scaling and learned projections, transformer attention.`,
        tests: [
          { name: 'zero scorer gives uniform weights', setup: 'H = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])\nWs = np.ones((4, 2)); Wh = np.ones((4, 2)); v = np.zeros(4)', call: 'additive_attention(np.array([0.5, -0.5]), H, Ws, Wh, v)', expect: '(np.full(3, 1 / 3), H.mean(axis=0))' },
          { name: 'random parameters vs reference', setup: 'rng = np.random.default_rng(0)\ns = rng.normal(size=3); H = rng.normal(size=(5, 4))\nWs = rng.normal(size=(6, 3)); Wh = rng.normal(size=(6, 4)); v = rng.normal(size=6)', call: 'additive_attention(s, H, Ws, Wh, v)' },
          { name: 'weights form a distribution, even with huge scores', setup: 'rng = np.random.default_rng(1)\ns = rng.normal(size=2); H = rng.normal(size=(7, 3))\nWs = rng.normal(size=(5, 2)); Wh = rng.normal(size=(5, 3)); v = 500 * rng.normal(size=5)', code: 'a, c = additive_attention(s, H, Ws, Wh, v)\nassert np.all(np.isfinite(a)) and np.isclose(a.sum(), 1)\nassert c.shape == (3,)', hidden: true },
          { name: 'another configuration vs reference', setup: 'rng = np.random.default_rng(2)\ns = rng.normal(size=4); H = rng.normal(size=(3, 2))\nWs = rng.normal(size=(3, 4)); Wh = rng.normal(size=(3, 2)); v = rng.normal(size=3)', call: 'additive_attention(s, H, Ws, Wh, v)', hidden: true }
        ]
      },
      {
        id: 'seq2seq-decoding', title: 'Teacher Forcing and Greedy Decoding', difficulty: 'easy',
        prompt: `Implement two pieces of a seq2seq pipeline:

- \`make_decoder_io(target, bos, eos)\` → \`(decoder_input, decoder_target)\`: the input is \`[bos] + target\` and the target is \`target + [eos]\`: at step t the decoder reads token t and must predict token t + 1.
- \`greedy_decode(step, state, bos, eos, max_len)\`: start from \`bos\`; repeat at most \`max_len\` times: \`logits, state = step(state, token)\`, choose \`token = int(np.argmax(logits))\`; stop **without** adding it when it equals \`eos\`, otherwise append it. Return the list of generated tokens.

\`\`\`text
make_decoder_io([5, 6], bos=0, eos=1)  ->  ([0, 5, 6], [5, 6, 1])
\`\`\``,
        starter: `
import numpy as np

def make_decoder_io(target, bos, eos):
    # TODO
    pass

def greedy_decode(step, state, bos, eos, max_len):
    tokens = []
    # TODO
    return tokens
`,
        solution: `
import numpy as np

def make_decoder_io(target, bos, eos):
    return [bos] + list(target), list(target) + [eos]

def greedy_decode(step, state, bos, eos, max_len):
    tokens, token = [], bos
    for _ in range(max_len):
        logits, state = step(state, token)
        token = int(np.argmax(logits))
        if token == eos:
            break
        tokens.append(token)
    return tokens
`,
        hints: ['Shifting by one position is all teacher forcing needs.', 'Feed the token you just chose back into `step` on the next iteration.'],
        explain: `Training and inference differ in exactly one way: during training the decoder input is the shifted ground truth; during inference it is the model's own previous choice. That mismatch is called exposure bias.`,
        tests: [
          { call: 'make_decoder_io([5, 6], 0, 1)', expect: '([0, 5, 6], [5, 6, 1])' },
          { name: 'decodes until <eos>', setup: 'def step(state, tok):\n    logits = np.zeros(6)\n    logits[[3, 4, 2, 5, 1][min(state, 4)]] = 1.0\n    return logits, state + 1', call: 'greedy_decode(step, 0, 0, 1, 10)', expect: '[3, 4, 2, 5]' },
          { name: 'stops at max_len', setup: 'def step(state, tok):\n    logits = np.zeros(4)\n    logits[2] = 1.0\n    return logits, state', call: 'greedy_decode(step, None, 0, 1, 3)', expect: '[2, 2, 2]', hidden: true },
          { name: 'the previous token is fed back', setup: 'def step(state, tok):\n    logits = np.zeros(10)\n    logits[(tok + 3) % 10] = 1.0\n    return logits, state', call: 'greedy_decode(step, 0, 0, 9, 8)', expect: '[3, 6]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Sutskever et al., Sequence to Sequence Learning with Neural Networks', url: 'https://arxiv.org/abs/1409.3215' },
      { title: 'Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate', url: 'https://arxiv.org/abs/1409.0473' },
      { title: 'Jay Alammar, Visualizing a Neural Machine Translation Model', url: 'https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/' }
    ]
  },

  // =====================================================================================
  {
    id: 'bert-encoders', branch: 'llm', title: 'Encoder Models: BERT & Masked Language Modeling', short: 'BERT & Encoders', level: 3, minutes: 40, prereqs: ['transformer-block', 'language-models'],
    tags: ['bert', 'encoder', 'masked language modeling', 'mlm', 'cls token', 'bidirectional', 'fine-tuning', 'classification', 'ner', 'roberta'],
    summary: 'The other half of the transformer family: bidirectional encoders such as BERT, pretrained by masking words and predicting them, then fine-tuned for classification, tagging and search. How masking works, the [CLS] token and task heads, and when an encoder beats a GPT.',
    why: `GPT-style decoders generate text, but many tasks are about **understanding**: classifying support tickets, finding names and dates in documents, matching questions to answers. Encoder models like BERT read the whole input at once (every token sees both its left and its right context) which makes them small, fast and excellent at these tasks. They are also the starting point of most text-embedding models used in search and RAG.`,
    objectives: [
      'Contrast bidirectional encoders with causal decoders',
      'Explain masked language modeling, including the 80/10/10 rule, and why it allows bidirectional context',
      'Use the [CLS] token and task heads for classification and token tagging',
      'Choose between an encoder, a decoder and an encoder–decoder for a task'
    ],
    sections: [
      { type: 'text', title: 'Bidirectional context', md: `A GPT predicts the next token, so each position may only see what came **before** it (the causal mask). But to understand "bank" in "I sat on the bank of the river", the words *after* it matter too.

An **encoder** runs self-attention **without** a causal mask: every token attends to every other token, left and right. The catch: you can no longer train it to predict the next word (it would simply look at it). Encoders need a different pretraining objective.` },
      { type: 'text', title: 'Masked language modeling', md: `BERT (2018) hides some tokens and asks the model to recover them from both sides. For each sequence, 15% of the token positions are selected; of those:

- 80% are replaced by a special [MASK] token,
- 10% are replaced by a random token,
- 10% are left unchanged.

The loss is the cross-entropy at the selected positions only:

$$ L = −(1 / |M|) · Σ_{i ∈ M} log p( x_i | x̃ ) $$
:::where
M — the set of selected positions; |M| is how many there are
x_i — the original token at position i
x̃ — the corrupted input sequence the model actually sees
p( x_i | x̃ ) — the probability the model gives to the original token at position i
:::

Why not always use [MASK]? That token never appears when the model is later fine-tuned on real text; the random and unchanged cases force it to build good representations for every token, not just masked ones. BERT also used a "next sentence prediction" task, which RoBERTa later showed was unnecessary. Successors: RoBERTa (more data, longer training), ELECTRA (detect replaced tokens, every position gives a training signal), DeBERTa, ModernBERT.` },
      { type: 'code', title: 'Masking a batch for MLM', code: `
import numpy as np
rng = np.random.default_rng(7)
vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "the", "cat", "sat", "on", "mat", "dog", "ran", "a", "park", "in"]
MASK, V = 3, len(vocab)
ids = np.array([1, 4, 5, 6, 7, 4, 8, 2, 11, 9, 10, 13, 4, 12, 2])           # [CLS] ... [SEP] ... [SEP]
special = ids <= 3                                                           # never mask special tokens

selected = (rng.random(len(ids)) < 0.3) & ~special                           # 30% here so the demo shows more
r = rng.random(len(ids))
random_tokens = rng.integers(4, V, len(ids))
inputs = ids.copy()
inputs[selected & (r < 0.8)] = MASK                                          # 80% -> [MASK]
swap = selected & (r >= 0.8) & (r < 0.9)
inputs[swap] = random_tokens[swap]                                           # 10% -> a random token
labels = np.where(selected, ids, -100)                                       # -100 = ignored by the loss

print("original:", " ".join(vocab[i] for i in ids))
print("input:   ", " ".join(vocab[i] for i in inputs))
print("predict: ", [vocab[i] for i in labels if i >= 0])
`, after: `Every selected position must be predicted, including any that kept their original token (the 10% case), so the model can't simply learn "only [MASK] needs a guess". Special tokens are never selected.` },
      { type: 'check', question: { type: 'mcq', q: 'Why can\'t an encoder like BERT be trained with plain next-token prediction?', options: ['It has no output layer', 'Without a causal mask every position can see the next token, so the task would be trivial', 'It is too small', 'It uses a different tokenizer'], answer: 1, explain: 'Bidirectional attention requires a different objective, such as masking.' } },
      { type: 'text', title: 'Special tokens and task heads', md: `BERT-style inputs look like \`[CLS] first text [SEP] second text [SEP]\`, and the encoder outputs one vector per token. Small **task heads** sit on top:

| Task | Head |
| --- | --- |
| Sentence classification (sentiment, topic, spam) | a linear layer on the final **[CLS]** vector (or on the mean of all token vectors) |
| Token classification (named entities, part of speech) | a linear layer on **every** token vector |
| Sentence pairs (entailment, duplicate questions) | both texts in one input, classify [CLS] |
| Extractive question answering | predict the start and end positions of the answer span |

**Fine-tuning** adds the head and trains the whole network with a small learning rate (≈ 2e-5) for a few epochs. A model with ~100M parameters, fine-tuned on a few thousand labelled examples, is often as accurate as a prompted LLM: at a tiny fraction of the cost and latency.` },
      { type: 'code', title: 'Fine-tuning a BERT-style classifier with Hugging Face', lang: 'pytorch', code: `
# pip install torch transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)   # adds a fresh classification head

texts = ["Great product, works perfectly", "Broke after two days", "Love it", "Terrible support"]
labels = torch.tensor([1, 0, 1, 0])
batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    out = model(**batch, labels=labels)              # cross-entropy on the classification head
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    print(epoch, round(out.loss.item(), 4))

model.eval()
with torch.no_grad():
    print(model(**tok(["Works great"], return_tensors="pt")).logits.softmax(-1))
` },
      { type: 'text', title: "The encoder family since BERT", md: `BERT's recipe was quickly refined:

| Model | Year | What changed |
| --- | --- | --- |
| RoBERTa | 2019 | trained longer on ten times more data, with fresh masks every epoch and no next-sentence prediction: the same architecture, much better results |
| ALBERT | 2019 | shared weights across layers and a factorised embedding to cut parameters |
| ELECTRA | 2020 | a small generator replaces some tokens and the encoder learns to spot which ones were replaced, getting a training signal from every position instead of only the 15% that are masked |
| DeBERTa | 2020 | attention that keeps content and relative position separate; strong results on understanding benchmarks |
| ModernBERT | 2024 | modern components (RoPE, gated feed-forward layers, efficient attention) and an 8,192-token context |

Encoders remain the workhorse for classification, extraction, embeddings and reranking: they are small, fast and cheap to fine-tune, and many text-embedding and reranking models are encoders underneath (see [Text Embedding Models & Semantic Search](#/concept/text-embeddings)).` },
      { type: 'code', title: "How many positions give a training signal?", code: `seq_len, batch = 512, 32
mlm_positions = round(0.15 * seq_len)                 # BERT computes the loss only on the selected 15%
causal_positions = seq_len                            # a GPT-style model is trained on every position
print(f"masked LM:  {mlm_positions} of {seq_len} positions per sequence, {mlm_positions * batch:,} per batch")
print(f"causal LM:  {causal_positions} of {seq_len} positions per sequence, {causal_positions * batch:,} per batch")
print(f"ELECTRA:    {seq_len} of {seq_len} positions per sequence (every token is classified as original or replaced)")
print("MLM learns from about 15% of the tokens it processes:", mlm_positions == 77)`, after: `Masked language modelling pays for a full forward pass over all 512 positions but learns from only 77 of them, one reason ELECTRA, which gets a signal from every token, trains more efficiently. Causal models also learn from every position, at the price of seeing only the context to the left.`, predict: {"type":"mcq","q":"In a 512-token sequence, on roughly how many positions does BERT's masked-language-modelling loss operate?","options":["About 8","About 77","About 256","All 512"],"answer":1,"explain":"15% of 512 positions are selected for prediction: about 77.","expect":["MLM learns from about 15% of the tokens it processes: True"]} },
      { type: 'text', title: 'Encoder, decoder or encoder–decoder?', md: `| Need | Typical choice | Why |
| --- | --- | --- |
| Classify, tag, extract, embed, rerank | **encoder** (BERT family) | bidirectional context, small, fast, cheap to fine-tune |
| Generate: chat, code, open-ended writing, reasoning | **decoder** (GPT family) | trained to continue text |
| Transform one text into another: translate, summarize, transcribe | **encoder–decoder** (T5, BART, Whisper) | separate reading and writing, linked by cross-attention |

Large decoders can do classification too (by prompting), and are the right tool when labelled data is scarce or the task changes often. For a fixed, high-volume task, a fine-tuned encoder is usually far cheaper and faster.` },
      { type: 'check', question: { type: 'tf', q: 'For sentence classification with BERT, a linear layer is usually applied to the final vector of the [CLS] token.', answer: true, explain: 'That vector attends to the whole input and is trained to summarize it.' } }
    ],
    when: `- Text classification, named-entity recognition, extraction and reranking at scale.
- As the backbone of embedding models for semantic search and RAG.
- When latency and cost matter more than generative flexibility.
- Prefer a prompted decoder when you have almost no labelled data or need free-form outputs.`,
    pitfalls: `- Using a raw pretrained BERT's [CLS] vector as a sentence embedding without contrastive fine-tuning: similarities are poor.
- Exceeding the maximum input length (often 512 tokens) and silently truncating important text.
- Too high a learning rate when fine-tuning (catastrophic forgetting); too many epochs on small data.
- Mismatched tokenizers between pretraining and fine-tuning.
- Masking special tokens or padding during MLM.`,
    takeaways: [
      'Encoders use bidirectional self-attention: every token sees the whole input.',
      'Masked language modeling: select 15% of positions; 80% [MASK], 10% random, 10% unchanged; predict the originals.',
      'Task heads: [CLS] vector for sentence tasks, every token vector for tagging.',
      'Fine-tuned encoders are small, fast and strong for understanding tasks.',
      'Encoder for understanding, decoder for generation, encoder–decoder for transformation.'
    ],
    quiz: [
      { type: 'fill', q: 'In BERT\'s masking scheme, what percentage of the selected positions is replaced by the [MASK] token?', answer: ['80', '80%'], explain: '80% [MASK], 10% random token, 10% unchanged.' },
      { type: 'mcq', q: 'Which task suits a BERT-style encoder best?', options: ['Writing a short story', 'Tagging every person and company name in a document', 'Holding a long conversation', 'Translating speech to text'], answer: 1, explain: 'Token classification needs per-token, bidirectional understanding.' },
      { type: 'tf', q: 'BERT is trained with a causal mask, like GPT.', answer: false, explain: 'It attends in both directions and is trained with masking instead.' },
      { type: 'mcq', q: 'Why does BERT sometimes leave a selected token unchanged?', options: ['To save compute', 'So the model learns useful representations for real tokens, which is what it sees after fine-tuning', 'It is a bug', 'To mark sentence boundaries'], answer: 1, explain: 'The [MASK] token never appears at fine-tuning time.' },
      { type: 'mcq', q: 'What is ELECTRA\'s pretraining task?', options: ['Next-sentence prediction', 'Detecting which tokens were replaced by a small generator', 'Translation', 'Image captioning'], answer: 1, explain: 'Every position gets a training signal, which makes it sample-efficient.' },
      {"type":"mcq","q":"What was the main lesson of RoBERTa?","options":["Encoders need causal masks","BERT was under-trained: more data, longer training and dynamic masking improve it substantially without changing the architecture","Next-sentence prediction is essential","Smaller vocabularies are better"],"answer":1,"explain":"RoBERTa kept BERT's architecture and showed that training choices, not architecture, limited it."},
      {"type":"tf","q":"Many text-embedding and reranking models are encoder (BERT-style) models underneath.","answer":true,"explain":"Bidirectional encoders produce strong representations of a whole text cheaply, which is what embedding and reranking need."}
    ],
    problems: [
      {
        id: 'mlm-masking', title: 'Masked Language Modeling Inputs', difficulty: 'medium',
        prompt: `Implement \`mask_tokens(ids, mask_id, vocab_size, rng, p=0.15, special_ids=())\` → \`(inputs, labels)\` as NumPy arrays. Draw from the generator **exactly** in this order (each a full-length array):

1. \`sel = rng.random(n) < p\`, then remove positions whose token is in \`special_ids\`;
2. \`r = rng.random(n)\`;
3. \`random_tokens = rng.integers(0, vocab_size, n)\`.

Then, starting from a copy of ids: selected positions with r < 0.8 become \`mask_id\`; selected positions with 0.8 ≤ r < 0.9 become \`random_tokens\` at that position; the remaining selected positions are unchanged. \`labels\` holds the original id at selected positions and −100 everywhere else.`,
        starter: `
import numpy as np

def mask_tokens(ids, mask_id, vocab_size, rng, p=0.15, special_ids=()):
    ids = np.asarray(ids)
    # TODO
    pass
`,
        solution: `
import numpy as np

def mask_tokens(ids, mask_id, vocab_size, rng, p=0.15, special_ids=()):
    ids = np.asarray(ids)
    n = len(ids)
    sel = rng.random(n) < p
    if len(special_ids):
        sel = sel & ~np.isin(ids, list(special_ids))
    r = rng.random(n)
    random_tokens = rng.integers(0, vocab_size, n)
    inputs = ids.copy()
    inputs[sel & (r < 0.8)] = mask_id
    swap = sel & (r >= 0.8) & (r < 0.9)
    inputs[swap] = random_tokens[swap]
    labels = np.where(sel, ids, -100)
    return inputs, labels
`,
        hints: ['Combine boolean masks with `&` and `~`.', '`np.isin(ids, list(special_ids))` marks special tokens.', '`np.where(sel, ids, -100)` builds the labels.'],
        explain: `Drawing full-length random arrays (rather than looping) keeps the procedure vectorized and reproducible. Only selected positions carry a label, so the loss ignores the other 85% of tokens: one reason MLM needs a lot of data.`,
        tests: [
          { name: 'reference run with a fixed seed', setup: 'ids = np.array([1, 5, 6, 7, 8, 9, 10, 11, 12, 2])', call: 'mask_tokens(ids, 3, 20, np.random.default_rng(0), p=0.5, special_ids=(1, 2))' },
          { name: 'labels only where something was selected', setup: 'ids = np.arange(4, 60)', code: 'inputs, labels = mask_tokens(ids, 3, 100, np.random.default_rng(1), p=0.3)\nchanged = inputs != ids\nassert np.all(labels[changed] == ids[changed]), "every changed token must have a label"\nassert np.all((labels == -100) | (labels == ids))' },
          { name: 'special tokens are never selected', setup: 'ids = np.array([1, 2] * 40)', code: 'inputs, labels = mask_tokens(ids, 3, 50, np.random.default_rng(2), p=0.9, special_ids=(1, 2))\nassert np.all(labels == -100) and np.array_equal(inputs, ids)', hidden: true },
          { name: 'proportions on a long sequence', setup: 'ids = np.random.default_rng(3).integers(5, 1000, 20000)', code: 'inputs, labels = mask_tokens(ids, 3, 1000, np.random.default_rng(4))\nsel = labels != -100\nassert 0.13 < sel.mean() < 0.17\nassert 0.75 < np.mean(inputs[sel] == 3) < 0.85', hidden: true }
        ]
      },
      {
        id: 'cls-pooling-head', title: 'Sentence Vectors and a Classification Head', difficulty: 'easy',
        prompt: `An encoder returned \`hidden\`, one vector per token (shape (T, d)), with position 0 being [CLS]. \`mask\` (length T) is 1 for real tokens and 0 for padding. Implement:

- \`cls_vector(hidden)\` → the [CLS] vector (row 0)
- \`mean_pool(hidden, mask)\` → the average of the rows where mask is 1
- \`classify(vector, W, b)\` → class probabilities: softmax(vector @ W + b), numerically stable

\`\`\`text
mean_pool([[1, 1], [3, 5], [9, 9]], [1, 1, 0])  ->  [2.0, 3.0]
\`\`\``,
        starter: `
import numpy as np

def cls_vector(hidden):
    # TODO
    pass

def mean_pool(hidden, mask):
    # TODO
    pass

def classify(vector, W, b):
    # TODO
    pass
`,
        solution: `
import numpy as np

def cls_vector(hidden):
    return np.asarray(hidden, dtype=float)[0]

def mean_pool(hidden, mask):
    hidden = np.asarray(hidden, dtype=float)
    m = np.asarray(mask, dtype=float)[:, None]
    return (hidden * m).sum(axis=0) / m.sum()

def classify(vector, W, b):
    z = np.asarray(vector, dtype=float) @ np.asarray(W, dtype=float) + np.asarray(b, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
`,
        hints: ['Multiply by the mask as a column (`mask[:, None]`) so padding rows become zero.', 'Divide by the number of real tokens, not by T.'],
        explain: `Mean pooling over real tokens is the usual way to turn an encoder's outputs into one sentence vector; the [CLS] vector is the classic alternative. Either one feeds a tiny linear head during fine-tuning.`,
        tests: [
          { call: 'mean_pool([[1, 1], [3, 5], [9, 9]], [1, 1, 0])', expect: 'np.array([2.0, 3.0])' },
          { call: 'cls_vector([[0.5, -1.0], [2.0, 2.0]])', expect: 'np.array([0.5, -1.0])' },
          { name: 'classifier probabilities', call: 'classify([1.0, 0.0], np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0]))', expect: 'np.exp([2.0, 0.0]) / np.exp([2.0, 0.0]).sum()' },
          { name: 'random encoder output vs reference', setup: 'rng = np.random.default_rng(0)\nH = rng.normal(size=(6, 4)); mask = np.array([1, 1, 1, 1, 0, 0]); W = rng.normal(size=(4, 3)); b = rng.normal(size=3)', call: '[mean_pool(H, mask), classify(cls_vector(H), W, b)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers', url: 'https://arxiv.org/abs/1810.04805' },
      { title: 'Liu et al., RoBERTa: A Robustly Optimized BERT Pretraining Approach', url: 'https://arxiv.org/abs/1907.11692' },
      { title: 'Jay Alammar, The Illustrated BERT', url: 'https://jalammar.github.io/illustrated-bert/' }
    ]
  },

  // =====================================================================================
  {
    id: 'text-embeddings', branch: 'llm', title: 'Text Embedding Models & Semantic Search', short: 'Text Embeddings', level: 3, minutes: 40, prereqs: ['bert-encoders', 'embeddings'],
    tags: ['sentence embeddings', 'semantic search', 'contrastive learning', 'infonce', 'in-batch negatives', 'hard negatives', 'bi-encoder', 'cross-encoder', 'mteb', 'pooling'],
    summary: 'How the sentence-embedding models behind search and RAG are built: pooling an encoder into one vector, contrastive training with in-batch and hard negatives, bi-encoders versus cross-encoders, and how to choose, evaluate and fine-tune an embedding model for your data.',
    why: `RAG, semantic search, deduplication, clustering and recommendations all start with an **embedding model** that turns text into a vector. Its quality caps everything downstream: if the right passage doesn't land near the question, no LLM can save the answer. Knowing how these models are trained explains their strengths and blind spots, and how to choose (or fine-tune) the right one for your data.`,
    objectives: [
      'Turn an encoder\'s token vectors into one normalized sentence vector by pooling',
      'Explain contrastive training with in-batch and hard negatives (the InfoNCE loss)',
      'Compare bi-encoders (fast retrieval) with cross-encoders (accurate reranking)',
      'Choose, evaluate and fine-tune an embedding model for a specific domain'
    ],
    sections: [
      { type: 'text', title: 'From token vectors to one sentence vector', md: `An encoder produces one vector per token. A **sentence embedding** needs one vector for the whole text, obtained by **pooling**:

$$ e = normalize( (1/n) Σ_{i=1..n} h_i ) $$
:::where
h_i — the encoder's output vector for token i (padding tokens are left out)
n — the number of real tokens in the text
normalize — divide by the vector's length, so every embedding has length 1 and cosine similarity becomes a dot product
e — the sentence embedding
:::

Mean pooling is the most common choice; the [CLS] vector or a learned attention-pooling layer are alternatives. A raw pretrained BERT, pooled this way, gives surprisingly **poor** similarities: its vectors all point in similar directions. Good embedding models are trained specifically so that similarity in vector space means similarity in meaning.` },
      { type: 'text', title: 'Contrastive training', md: `Training data consists of **pairs that belong together**: a search query and the page that was clicked, a question and its answer, a title and its article, a sentence and its paraphrase, or synthetic questions generated by an LLM for each passage. In a batch of B pairs, each query's positive is its own passage, and the other B − 1 passages serve as **in-batch negatives**:

$$ L = −(1/B) Σ_i log [ exp( sim(q_i, p_i) / τ ) / Σ_j exp( sim(q_i, p_j) / τ ) ] $$
:::where
q_i, p_i — the embeddings of query i and its matching (positive) passage
sim — cosine similarity
τ — "tau": a temperature, typically about 0.05, that sharpens the softmax
Σ_j — a sum over all passages in the batch; every p_j with j ≠ i acts as a negative
:::

This **InfoNCE** loss is a softmax classification: "which of these B passages answers my query?": the same idea as CLIP's image–text loss. Two levers matter most:
- **Bigger batches** give more negatives per step.
- **Hard negatives**: passages that look relevant but aren't, mined with keyword search or an earlier model (teach fine distinctions that random negatives never require).` },
      { type: 'code', title: 'In-batch contrastive loss', code: `
import numpy as np
rng = np.random.default_rng(0)

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def info_nce(q, candidates, tau=0.05):
    logits = normalize(q) @ normalize(candidates).T / tau      # every query against every candidate
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-np.mean(logp[np.arange(len(q)), np.arange(len(q))]))   # the positive for query i is candidate i

B, d = 16, 64
queries = rng.normal(size=(B, d))
positives = queries + 0.5 * rng.normal(size=(B, d))             # matching passages
hard = queries + 0.7 * rng.normal(size=(B, d))                  # look-alike passages that are NOT answers
print("a uniform guess scores ln B =", round(float(np.log(B)), 3))
print("random 'passages':        ", round(info_nce(queries, rng.normal(size=(B, d))), 3))
print("true positives:           ", round(info_nce(queries, positives), 3))
print("positives + hard negatives:", round(info_nce(queries, np.vstack([positives, hard])), 3))
`, after: `With true positives the loss is near zero: random passages are easy to rule out. Adding look-alike hard negatives makes the task harder and the loss larger: that difficulty is exactly what pushes a model to learn finer distinctions.` },
      { type: 'check', question: { type: 'mcq', q: 'What are in-batch negatives?', options: ['Tokens with negative embeddings', 'The other pairs\' passages in the same batch, used as wrong answers for each query', 'Examples with negative labels from humans', 'Gradients below zero'], answer: 1, explain: 'They come for free: B pairs give B − 1 negatives per query.' } },
      { type: 'text', title: 'Bi-encoders and cross-encoders', md: `| | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| How it scores | embeds query and document **separately**, compares vectors | reads query and document **together** in one transformer pass |
| Speed | document vectors precomputed; search millions in milliseconds | one full forward pass per (query, document) pair |
| Accuracy | good | better: full attention between the two texts |
| Typical role | first-stage **retrieval** | **reranking** the top 20–100 candidates |

Late-interaction models such as ColBERT keep one vector per token and sit in between. A strong RAG pipeline usually combines them: hybrid retrieval (keywords + bi-encoder), then a cross-encoder reranker.` },
      { type: 'text', title: 'Choosing, evaluating and fine-tuning', md: `- **Benchmarks**: the MTEB leaderboard compares models by task (retrieval, clustering, classification, reranking) and language. Look at the *retrieval* scores for your languages.
- **Practical constraints**: vector size (storage and speed), maximum input length, inference cost, licence.
- **Evaluate on your own data**: collect 50–200 (question, relevant passage) pairs and measure recall@k and MRR (see *RAG*). Benchmark winners don't always win on legal, medical or code text.
- **Tricks**: some models expect prefixes such as "query: " and "passage: "; normalize vectors; *Matryoshka* models let you truncate vectors to fewer dimensions with little loss; int8 or binary quantization cuts storage by 4 to 32 times.
- **Fine-tune** when your domain differs: a few thousand in-domain pairs plus mined hard negatives often give large gains.
- Always embed queries and documents with the **same** model, and re-embed everything when you change models.` },
      { type: 'text', title: "Vector size, precision and the cost of an index", md: `An index of embeddings costs memory in proportion to the number of vectors, their dimension and the bytes per number:

$$ index size = N · d · b $$
:::where
N — the number of stored vectors (documents or chunks)
d — the embedding dimension, typically 384 to 3,072
b — bytes per number: 4 for float32, 2 for float16, 1 for int8, 1/8 for binary (one bit per dimension)
:::

Ten million chunks of 1,024-dimensional float32 vectors take 41 GB, more than fits comfortably in one machine's memory. Three standard ways to shrink an index:
- **lower precision**: int8 quantization (4 times smaller) loses little; binary quantization (32 times smaller) loses more, so it is usually followed by rescoring a short list with the full vectors;
- **fewer dimensions**: models trained with Matryoshka-style objectives pack the most important information into the first dimensions, so their vectors can be truncated (say from 1,024 to 256) with a modest loss;
- **compression inside the index**: product quantization in approximate-nearest-neighbour indexes (see [Vector Search](#/concept/vector-search)).

The chart below shows the dimension trade-off on a small real example.` },
      { type: 'chart', title: "Finding the right lesson from its one-line summary", md: `A real retrieval test on this app: each lesson's one-sentence summary is the query, and the system must rank that lesson's full text first among all 106 lessons:`, after: `Sparse word-count vectors find the right lesson for 98% of the summaries, using 4,080 dimensions. Compressing them into dense vectors with an SVD costs a lot at 8 dimensions (70%) but reaches the sparse vectors' accuracy by 64: a few dozen directions capture most of what distinguishes the lessons. Modern embedding models work the same way in spirit, with learned instead of SVD directions, and models trained with nested (Matryoshka) objectives let you truncate their vectors to trade accuracy for index size.`, chart: {"kind":"line","x":[8,16,32,64,96],"xLog":true,"series":[{"name":"dense vectors from a truncated SVD of TF-IDF (LSA)","y":[0.698,0.906,0.972,0.981,0.981]}],"xLabel":"vector dimension (log scale)","yLabel":"top-1 accuracy (106 lessons)","yMin":0,"yMax":1,"hlines":[{"y":0.981,"label":"sparse TF-IDF, 4,080 dimensions"}]} },
      { type: 'code', title: "Sizing an embedding index", code: `def index_gb(n_vectors, dim, bytes_per_number):
    return n_vectors * dim * bytes_per_number / 1e9

n = 10_000_000                                   # ten million chunks
for dim in (1024, 256):
    for name, b in [("float32", 4), ("int8", 1), ("binary", 1 / 8)]:
        print(f"d = {dim:5d}, {name:<7}: {index_gb(n, dim, b):7.2f} GB")
print("binary is 32 times smaller than float32:", index_gb(n, 1024, 4) / index_gb(n, 1024, 1 / 8) == 32)`, after: `The 41 GB of float32 vectors shrink to about 10 GB in int8 and 1.3 GB in binary, and truncating to 256 dimensions divides each figure by four again: binary vectors of 256 dimensions hold ten million chunks in about 320 MB.`, predict: {"type":"mcq","q":"How large is an index of 10 million 1,024-dimensional float32 vectors?","options":["About 4 GB","About 41 GB","About 410 GB","About 0.4 GB"],"answer":1,"explain":"10,000,000 × 1,024 × 4 bytes ≈ 41 billion bytes = 41 GB.","expect":["binary is 32 times smaller than float32: True"]} },
      { type: 'code', title: 'Search and rerank with sentence-transformers', run: false, md: `Run locally after \`pip install sentence-transformers\`. The bi-encoder retrieves candidates; the cross-encoder reranks them.`, code: `
from sentence_transformers import SentenceTransformer, CrossEncoder

docs = ["Opened electronics can be returned within 14 days.",
        "Standard shipping takes 3 to 5 business days.",
        "All electronics include a two-year warranty against defects.",
        "Sale items can be exchanged but not refunded."]
query = "Can I send back a laptop I already unboxed?"

bi = SentenceTransformer("all-MiniLM-L6-v2")
D = bi.encode(docs, normalize_embeddings=True)
q = bi.encode([query], normalize_embeddings=True)[0]
candidates = (D @ q).argsort()[::-1][:3]                       # fast first stage

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, docs[i]) for i in candidates])
for i, s in sorted(zip(candidates, scores), key=lambda t: -t[1]):
    print(f"{s:6.2f}  {docs[i]}")
` },
      { type: 'check', question: { type: 'tf', q: 'A cross-encoder can precompute document vectors once and reuse them for every query.', answer: false, explain: 'It reads query and document together, so it must run once per pair; that is why it is used for reranking.' } }
    ],
    when: `- Semantic search, RAG retrieval, deduplication, clustering, recommendations and classification with few labels.
- A bi-encoder for the first stage, a cross-encoder to rerank when accuracy matters.
- Fine-tune on in-domain pairs when off-the-shelf models miss your vocabulary.`,
    pitfalls: `- Mixing embeddings from different models (or versions) in one index.
- Forgetting required prefixes or normalization.
- Choosing a model only by overall leaderboard rank instead of retrieval quality on your data.
- Embedding chunks longer than the model's maximum input (silently truncated).
- Training with "negatives" that are actually relevant (false negatives): they teach the wrong lesson.`,
    takeaways: [
      'Sentence embedding = pool token vectors (usually the mean) and normalize.',
      'Contrastive InfoNCE training: pull each query towards its positive, away from in-batch and hard negatives.',
      'Bigger batches and hard negatives make better embedding models.',
      'Bi-encoders retrieve fast; cross-encoders rerank accurately.',
      'Evaluate on your own (query, passage) pairs and fine-tune when the domain differs.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why are sentence embeddings usually normalized to length 1?', options: ['To save memory', 'So that the dot product equals cosine similarity', 'To make training faster', 'Normalization is required by tokenizers'], answer: 1, explain: 'Then a single matrix product scores every document.' },
      { type: 'fill', q: 'A contrastive batch has 64 (query, passage) pairs. How many in-batch negatives does each query get?', answer: ['63'], explain: 'All other passages in the batch: 64 − 1.' },
      { type: 'mcq', q: 'What is a hard negative?', options: ['A passage with many tokens', 'A passage that looks relevant to the query but does not answer it', 'A negative number in an embedding', 'A query with no answer'], answer: 1, explain: 'It forces the model to learn fine distinctions.' },
      { type: 'tf', q: 'Bi-encoders are typically used for fast first-stage retrieval and cross-encoders for reranking.', answer: true, explain: 'Speed first, accuracy second.' },
      { type: 'mcq', q: 'You switch to a new embedding model. What must you do with the existing vector index?', options: ['Nothing', 'Re-embed all documents with the new model', 'Only re-embed new documents', 'Normalize the old vectors again'], answer: 1, explain: 'Vectors from different models live in different spaces.' },
      {"type":"mcq","q":"What does Matryoshka-style training allow?","options":["Larger vectors","Truncating vectors to fewer dimensions with only a modest loss in quality","Training without negatives","Searching without an index"],"answer":1,"explain":"The training objective makes every prefix of the vector a usable embedding, so you can trade accuracy for index size by keeping fewer dimensions."},
      {"type":"fill","q":"How many gigabytes do 1 million 768-dimensional float32 vectors take?","answer":["3.072","3.07","3.1"],"explain":"1,000,000 × 768 × 4 bytes = 3.072 GB."}
    ],
    problems: [
      {
        id: 'mean-pool-normalize', title: 'Pooling and Cosine Similarity', difficulty: 'easy',
        prompt: `Implement for a batch:

- \`mean_pool(token_vecs, mask)\` → (B, d): token_vecs has shape (B, T, d) and mask (B, T) is 1 for real tokens and 0 for padding. Average only the real tokens of each sequence; a sequence with no real tokens gives a zero vector.
- \`normalize(E)\` → each row divided by its length (zero rows stay zero).
- \`cosine_matrix(A, B)\` → the (len(A), len(B)) matrix of cosine similarities between the rows.

\`\`\`text
mean_pool([[[1, 1], [3, 5], [0, 0]]], [[1, 1, 0]])  ->  [[2.0, 3.0]]
\`\`\``,
        starter: `
import numpy as np

def mean_pool(token_vecs, mask):
    # TODO
    pass

def normalize(E):
    # TODO
    pass

def cosine_matrix(A, B):
    # TODO
    pass
`,
        solution: `
import numpy as np

def mean_pool(token_vecs, mask):
    X = np.asarray(token_vecs, dtype=float)
    m = np.asarray(mask, dtype=float)[:, :, None]
    counts = m.sum(axis=1)
    return (X * m).sum(axis=1) / np.maximum(counts, 1)

def normalize(E):
    E = np.asarray(E, dtype=float)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(n == 0, 1, n)

def cosine_matrix(A, B):
    return normalize(A) @ normalize(B).T
`,
        hints: ['Give the mask a trailing axis (`mask[:, :, None]`) so it broadcasts over d.', 'Guard the division with `np.maximum(counts, 1)`.', 'Normalize both sides, then one matrix product gives every cosine.'],
        explain: `These three functions are the inference-time core of every semantic search system: encode, pool, normalize, then score all documents with a single matrix product.`,
        tests: [
          { call: 'mean_pool([[[1, 1], [3, 5], [0, 0]]], [[1, 1, 0]])', expect: 'np.array([[2.0, 3.0]])' },
          { call: 'cosine_matrix([[1, 0], [1, 1]], [[2, 0], [0, 3]])', expect: 'np.array([[1.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]])' },
          { name: 'empty sequence and zero rows', call: '[mean_pool(np.ones((2, 3, 2)), [[0, 0, 0], [1, 0, 1]]), normalize([[0.0, 0.0], [3.0, 4.0]])]', expect: '[np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [0.6, 0.8]])]', hidden: true },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(3, 5, 4)); mask = (rng.random((3, 5)) < 0.7).astype(int); mask[:, 0] = 1', call: 'cosine_matrix(mean_pool(X, mask), mean_pool(X, mask))', hidden: true }
        ]
      },
      {
        id: 'info-nce', title: 'The InfoNCE Loss', difficulty: 'medium',
        prompt: `Implement \`info_nce(q, p, tau=0.05, hard=None)\` → a float. \`q\` and \`p\` are (B, d) arrays where row i of p is the positive for query i; \`hard\` is an optional (H, d) array of extra negatives.

1. Normalize every row of q, p (and hard) to unit length.
2. The candidates are the rows of p, followed by the rows of hard if given.
3. logits = q @ candidatesᵀ / tau, shape (B, number of candidates).
4. Return the mean over i of −log softmax(logits[i])[i], with a numerically stable log-softmax.

\`\`\`text
If every query and passage is identical, all candidates tie and the loss is ln(number of candidates).
\`\`\``,
        starter: `
import numpy as np

def info_nce(q, p, tau=0.05, hard=None):
    # TODO
    pass
`,
        solution: `
import numpy as np

def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def info_nce(q, p, tau=0.05, hard=None):
    q = _unit(q)
    cands = _unit(p) if hard is None else np.vstack([_unit(p), _unit(hard)])
    logits = q @ cands.T / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    i = np.arange(len(q))
    return float(-np.mean(logp[i, i]))
`,
        hints: ['`np.vstack([p, hard])` appends the extra negatives after the positives.', 'The correct candidate for query i sits in column i.', '`logp[np.arange(B), np.arange(B)]` picks those entries.'],
        explain: `InfoNCE is cross-entropy over "which candidate is mine?". The temperature controls how sharply small similarity differences are rewarded; hard negatives raise the loss until the model learns to separate look-alikes.`,
        tests: [
          { name: 'identical vectors give ln(candidates)', call: '[info_nce(np.ones((4, 3)), np.ones((4, 3))), info_nce(np.ones((4, 3)), np.ones((4, 3)), hard=np.ones((2, 3)))]', expect: '[np.log(4), np.log(6)]' },
          { name: 'orthogonal pairs are nearly perfect', call: 'info_nce(np.eye(3), 2 * np.eye(3), tau=0.1)', expect: 'float(-np.log(np.exp(10) / (np.exp(10) + 2)))' },
          { name: 'random batch with hard negatives vs reference', setup: 'rng = np.random.default_rng(0)\nq = rng.normal(size=(6, 8)); p = q + 0.3 * rng.normal(size=(6, 8)); hard = q + 0.6 * rng.normal(size=(6, 8))', call: '[info_nce(q, p), info_nce(q, p, tau=0.1, hard=hard)]', hidden: true },
          { name: 'hard negatives make it harder', setup: 'rng = np.random.default_rng(1)\nq = rng.normal(size=(10, 16)); p = q + 0.4 * rng.normal(size=(10, 16))', code: 'easy = info_nce(q, p, hard=rng.normal(size=(10, 16)))\nhard = info_nce(q, p, hard=q + 0.5 * rng.normal(size=(10, 16)))\nassert hard > easy', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Reimers & Gurevych, Sentence-BERT', url: 'https://arxiv.org/abs/1908.10084' },
      { title: 'Karpukhin et al., Dense Passage Retrieval for Open-Domain QA', url: 'https://arxiv.org/abs/2004.04906' },
      { title: 'MTEB, Massive Text Embedding Benchmark leaderboard', url: 'https://huggingface.co/spaces/mteb/leaderboard' },
      { title: 'Sentence Transformers documentation', url: 'https://www.sbert.net/' }
    ]
  },

  // =====================================================================================
  {
    id: 'training-at-scale', branch: 'llm', title: 'Training at Scale: GPUs, Mixed Precision & Parallelism', short: 'Training at Scale', level: 3, minutes: 50, prereqs: ['pretraining', 'pytorch'],
    tags: ['gpu', 'training memory', 'mixed precision', 'bf16', 'gradient accumulation', 'activation checkpointing', 'data parallel', 'ddp', 'fsdp', 'zero', 'tensor parallel', 'pipeline parallel', 'mfu'],
    summary: 'What it takes to train big models: where GPU memory goes (weights, gradients, optimizer states, activations), mixed precision, gradient accumulation and activation checkpointing, the parallelism strategies, data, sharded (ZeRO/FSDP), tensor and pipeline, and measuring efficiency with MFU.',
    why: `A 7B-parameter model doesn't fit on one GPU for training, and even a 124M model trains several times faster with the right tricks. Scaling up is mostly **memory and communication engineering**: knowing where every byte goes tells you which technique to reach for (bf16, gradient accumulation, checkpointing, sharding) and lets you estimate how many GPUs and how much time a run needs. This is the practical heart of "how to build an LLM".`,
    objectives: [
      'Compute training memory: weights, gradients, optimizer states and activations',
      'Use mixed precision, gradient accumulation and activation checkpointing, and know their trade-offs',
      'Explain data parallelism, ZeRO/FSDP sharding, tensor and pipeline parallelism',
      'Estimate throughput with MFU and choose a strategy for a given model and cluster'
    ],
    sections: [
      { type: 'text', title: 'Where the memory goes', md: `Training with Adam in mixed precision keeps, for every parameter, a bf16 weight, a bf16 gradient, an fp32 master copy of the weight and Adam's two fp32 moment estimates. On top of that come the **activations**: intermediate results saved during the forward pass because the backward pass needs them.

$$ memory ≈ 16 · N + activations $$
:::where
N — the number of model parameters
16 — bytes per parameter: 2 (bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master weight) + 8 (Adam's two fp32 moments)
activations — grow with batch size × sequence length × number of layers × width (and with sequence length squared for naive attention)
:::

A 7B model needs ≈ 112 GB just for its parameter-related state: more than one 80 GB GPU before a single activation is stored. The techniques below attack each term.` },
      { type: 'text', title: 'Mixed precision', md: `GPUs multiply 16-bit numbers many times faster than 32-bit ones. **Mixed precision** runs the forward and backward passes in **bf16** (or fp16) while keeping an fp32 master copy of the weights for the optimizer update, so tiny updates aren't rounded away.

- **bf16** has the same exponent range as fp32, so values rarely overflow or underflow: the modern default.
- **fp16** has more mantissa precision but a small range; it needs **loss scaling** (multiply the loss before backward, divide the gradients after) to keep small gradients from becoming zero.
- The newest GPUs also support **fp8** matrix multiplies with per-tensor scaling for even faster training.

Benefits: roughly half the activation memory and much higher throughput. In PyTorch it is one context manager: \`torch.autocast(device_type="cuda", dtype=torch.bfloat16)\`.` },
      { type: 'text', title: 'Gradient accumulation and activation checkpointing', md: `**Gradient accumulation**: if the batch you want doesn't fit, run several smaller micro-batches, add up their gradients, and take one optimizer step. Mathematically identical to the big batch (for losses that average over examples):

$$ effective batch = micro_batch · accumulation_steps · n_gpus $$
:::where
micro_batch — the examples processed at once on one GPU
accumulation_steps — how many micro-batches are summed before each optimizer step
n_gpus — the number of data-parallel GPUs
:::

**Activation checkpointing** (rematerialization): store only some activations (for example at the boundary of each transformer block) and recompute the rest during the backward pass. It costs roughly 30% extra compute and saves most of the activation memory, which often makes the difference between fitting and not fitting.` },
      { type: 'code', title: 'A training memory calculator', code: `
def gib(x):
    return x / 2 ** 30

def state_bytes(n_params):
    return 16 * n_params                          # bf16 weights + bf16 grads + fp32 master + Adam m, v

for name, n in [("GPT-2 small", 124e6), ("Llama 3 8B", 8e9), ("70B model", 70e9)]:
    total = state_bytes(n)
    print(f"{name:12s} parameter state {gib(total):8.1f} GiB  |  sharded over 8 GPUs {gib(total / 8):6.1f} GiB each"
          f"  |  over 64 GPUs {gib(total / 64):5.1f} GiB each")

micro, accum, gpus, seq_len = 8, 16, 8, 2048
batch = micro * accum * gpus
print(f"effective batch = {micro} x {accum} x {gpus} = {batch} sequences = {batch * seq_len / 1e6:.1f}M tokens per optimizer step")

n_params, tokens_per_sec, n_gpus, peak = 8e9, 2.0e6, 256, 989e12    # H100 bf16 dense peak ~ 989 TFLOP/s
print(f"MFU = 6 x N x tokens/s / (GPUs x peak) = {6 * n_params * tokens_per_sec / (n_gpus * peak):.1%}")
` },
      { type: 'check', question: { type: 'fill', q: 'Micro-batch 4, 8 accumulation steps, 16 GPUs: what is the effective batch size (in sequences)?', answer: ['512'], explain: '4 × 8 × 16 = 512.' } },
      { type: 'text', title: 'Data parallelism and sharding', md: `**Data parallelism (DDP)**: every GPU holds a full copy of the model and processes a different slice of the batch; after the backward pass the gradients are averaged across GPUs (an *all-reduce*), so all copies stay identical. Easy and efficient, but each GPU still needs the full 16·N bytes.

**ZeRO / FSDP** (fully sharded data parallel) removes that redundancy by splitting the state across the G GPUs:
- stage 1 shards the optimizer states, stage 2 also the gradients, stage 3 also the parameters;
- each layer's parameters are gathered just in time for its forward and backward pass, then freed.

$$ per-GPU state memory (ZeRO-3) ≈ 16 · N / G $$
:::where
N — the number of parameters
G — the number of GPUs sharing the state
:::

The price is more communication, which fast interconnects (NVLink, InfiniBand) make affordable.` },
      { type: 'chart', title: "Memory per GPU for a 7B model on 8 GPUs", md: `Mixed-precision Adam: 2 bytes of bf16 weights, 2 of gradients and 12 of fp32 optimizer state (master weights and two moments) per parameter:`, after: `Plain data parallelism keeps a full copy of all 16 bytes per parameter on every GPU: about 108 GB, more than an 80 GB GPU holds. Sharding the optimizer states across the 8 GPUs (ZeRO-1) brings that to 37 GB, sharding the gradients too (ZeRO-2) to 25 GB, and sharding the weights as well (ZeRO-3, PyTorch's FSDP) to 13.5 GB, at the price of gathering each layer's weights over the network just before it runs.`, chart: {"kind":"bar","x":["data parallel","ZeRO-1","ZeRO-2","ZeRO-3 / FSDP"],"series":[{"name":"weights + gradients + Adam states per GPU (GB, before activations)","y":[107.8,37.1,25.3,13.5]}],"xLabel":"strategy","yLabel":"GB per GPU","yMin":0} },
      { type: 'text', title: "Communication: the hidden cost", md: `Every data-parallel step ends with an **all-reduce**: each GPU contributes its gradients and receives the average. With the standard ring algorithm, each GPU sends and receives about twice the size of the gradients, whatever the number of GPUs:

$$ all-reduce time ≈ 2 · (G − 1) / G · gradient bytes / bandwidth $$
:::where
G — the number of GPUs taking part
gradient bytes — the size of all the gradients, for example 2 bytes per parameter in bf16
bandwidth — each GPU's network bandwidth in bytes per second: hundreds of GB/s over NVLink inside a server, around 50 GB/s (400 Gb/s) per GPU over InfiniBand between servers
:::

For a 7B model that means 14 GB of gradients every step: a fraction of a second inside one server, noticeably longer between servers. Training systems hide the cost by **overlapping**: the gradients of the last layers are sent while the backward pass is still computing the earlier ones, and small tensors are grouped into larger messages. Tensor parallelism communicates inside every layer, which is why it is kept within a server, while data and pipeline parallelism span servers.` },
      { type: 'code', title: "How long does an all-reduce take?", code: `def allreduce_seconds(n_params, gpus, bandwidth_gb_s, bytes_per_grad=2):
    grad_bytes = n_params * bytes_per_grad
    return 2 * (gpus - 1) / gpus * grad_bytes / (bandwidth_gb_s * 1e9)

n = 7e9                                          # a 7B model with bf16 gradients (14 GB)
inside = allreduce_seconds(n, 8, 450)            # 8 GPUs in one server, NVLink
across = allreduce_seconds(n, 64, 50)            # 64 GPUs in 8 servers, InfiniBand
print(f"8 GPUs over NVLink (~450 GB/s each):      {inside * 1000:6.0f} ms per step")
print(f"64 GPUs over InfiniBand (~50 GB/s each):  {across * 1000:6.0f} ms per step")
print(f"ratio: {across / inside:.1f}")
print("crossing servers is about 10 times slower:", 8 < across / inside < 12)`, after: `Inside one server the all-reduce takes about 55 milliseconds; across 64 GPUs on InfiniBand about half a second, ten times longer, because the network is about nine times slower while each GPU still moves roughly twice the gradient size. When a training step takes about a second, that is a heavy tax unless it is overlapped with the backward pass, which is exactly what training frameworks do.`, predict: {"type":"mcq","q":"Roughly how long does an all-reduce of 14 GB of bf16 gradients take across 64 GPUs connected at about 50 GB/s each?","options":["About 5 ms","About 0.5 s","About 5 s","About 50 s"],"answer":1,"explain":"Each GPU moves about 2 × 63/64 × 14 GB ≈ 27.6 GB at 50 GB/s: a little over half a second.","expect":["crossing servers is about 10 times slower: True"]} },
      { type: 'text', title: 'Model parallelism: tensor and pipeline', md: `When a single layer is too big, or to reduce communication at very large scale, the model itself is split:

| Strategy | What is split | Main cost | Typical use |
| --- | --- | --- | --- |
| **Data parallel (DDP)** | the batch | gradient all-reduce | always: the outer layer |
| **Sharded (ZeRO / FSDP)** | optimizer states, gradients, parameters | gathering parameters per layer | models whose state doesn't fit on one GPU |
| **Tensor parallel** | each matrix multiply (e.g. attention heads, FFN columns) | communication inside every layer: needs NVLink within a node | very wide layers |
| **Pipeline parallel** | consecutive layers on different GPUs | idle "bubbles" while the pipeline fills | very deep models across nodes |
| **Sequence / context parallel** | the sequence dimension | attention communication | very long contexts |

Frontier runs combine them ("3D parallelism" in Megatron-LM: tensor within a node, pipeline across nodes, data parallel across replicas) and add **expert parallelism** for mixture-of-experts models.` },
      { type: 'demo', title: 'Diagram: data, tensor and pipeline parallelism', demo: 'dg-parallel' },
      { type: 'text', title: 'Throughput and MFU', md: `Speed is measured in tokens per second, and efficiency as **model FLOPs utilization**:

$$ MFU = 6 · N · tokens_per_second / ( n_gpus · peak_flops ) $$
:::where
6 · N — the training FLOPs per token (forward plus backward) for an N-parameter model
tokens_per_second — the measured training throughput
peak_flops — one GPU's theoretical peak (about 10¹⁵ bf16 FLOP/s for an H100)
:::

Well-tuned large runs reach roughly 35–60%. A practical checklist: bf16 autocast, FlashAttention, \`torch.compile\`, the largest micro-batch that fits plus accumulation for the target batch size, enough data-loader workers, frequent checkpoints, and alerts on loss spikes and hardware failures (at thousands of GPUs, something fails every day).` },
      { type: 'code', title: 'Mixed precision and gradient accumulation in PyTorch', lang: 'pytorch', code: `
import torch

def train(model, loader, steps, accum_steps=8, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    for step, (x, y) in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # bf16 compute, fp32 master weights
            _, loss = model(x.cuda(), y.cuda())
        (loss / accum_steps).backward()                                   # gradients add up over micro-batches
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step + 1 == steps * accum_steps:
            break

# Data parallel across GPUs: wrap the model, launch one process per GPU with torchrun
#   model = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])
#   torchrun --nproc_per_node=8 train.py
# Fully sharded (ZeRO-3 style): torch.distributed.fsdp.FullyShardedDataParallel(model)
# Activation checkpointing for one block: torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
` },
      { type: 'check', question: { type: 'tf', q: 'Plain data parallelism (DDP) reduces the memory each GPU needs for the model\'s parameters and optimizer states.', answer: false, explain: 'Every GPU keeps a full copy; sharding (ZeRO/FSDP) is what splits the state.' } }
    ],
    when: `- Any training run that doesn't fit, or is too slow, on one GPU.
- bf16 mixed precision and FlashAttention: essentially always on modern GPUs.
- Gradient accumulation when the target batch size doesn't fit in memory.
- FSDP/ZeRO when the parameter state doesn't fit; tensor and pipeline parallelism at the largest scales.
- For fine-tuning, LoRA/QLoRA often removes the need for multi-GPU setups entirely.`,
    pitfalls: `- Forgetting that activations, not just weights, often dominate memory at long sequence lengths.
- Averaging micro-batch losses incorrectly with uneven micro-batch sizes.
- Using fp16 without loss scaling (gradients underflow to zero).
- Tensor parallelism across slow network links.
- Not checkpointing often enough: hardware failures are routine at scale.`,
    takeaways: [
      'Training state ≈ 16 bytes per parameter with Adam in mixed precision, plus activations.',
      'bf16 mixed precision: fast 16-bit compute with fp32 master weights.',
      'Gradient accumulation trades time for memory; checkpointing trades compute for memory.',
      'DDP replicates the model; ZeRO/FSDP shards it: ≈ 16·N/G bytes per GPU.',
      'Tensor and pipeline parallelism split the model itself; MFU measures how well the hardware is used.'
    ],
    quiz: [
      { type: 'fill', q: 'About how many GB does the parameter state (16 bytes per parameter) of a 13B model take?', answer: ['208', '208 gb', '208gb'], explain: '13·10⁹ × 16 bytes = 208 GB.' },
      { type: 'mcq', q: 'Why keep an fp32 master copy of the weights in mixed-precision training?', options: ['GPUs require it', 'Small updates would be rounded away in 16-bit weights', 'It doubles the speed', 'For checkpointing only'], answer: 1, explain: 'lr × gradient can be smaller than bf16 can represent relative to the weight.' },
      { type: 'mcq', q: 'What does ZeRO stage 3 / FSDP shard across GPUs?', options: ['Only the data', 'Optimizer states, gradients and parameters', 'Only the activations', 'Nothing: it replicates everything'], answer: 1, explain: 'Each GPU holds roughly 1/G of the state.' },
      { type: 'tf', q: 'Activation checkpointing reduces memory by recomputing some activations during the backward pass.', answer: true, explain: 'About 30% more compute for much less memory.' },
      { type: 'mcq', q: 'Which parallelism strategy needs very fast communication inside every layer?', options: ['Data parallelism', 'Tensor parallelism', 'Gradient accumulation', 'Pipeline parallelism'], answer: 1, explain: 'Each matrix multiply is split, so GPUs exchange partial results constantly.' },
      {"type":"mcq","q":"Which ZeRO stage also shards the model weights across GPUs?","options":["ZeRO-1","ZeRO-2","ZeRO-3 (FSDP)","None of them"],"answer":2,"explain":"ZeRO-1 shards optimizer states, ZeRO-2 adds gradients, and ZeRO-3 (PyTorch's FSDP) shards the weights as well, gathering each layer just before it runs."},
      {"type":"tf","q":"With a ring all-reduce, the data each GPU sends grows roughly in proportion to the number of GPUs.","answer":false,"explain":"Each GPU sends about 2·(G − 1)/G times the gradient size, which approaches twice the gradient size however many GPUs take part."}
    ],
    problems: [
      {
        id: 'training-memory', title: 'Training Memory and MFU', difficulty: 'easy',
        prompt: `Implement:

- \`state_bytes(n_params, bytes_per_param=16)\` → the parameter-related training state in bytes
- \`zero3_bytes_per_gpu(n_params, n_gpus, bytes_per_param=16)\` → that state shared evenly across the GPUs
- \`min_gpus(n_params, gpu_bytes, activation_bytes, bytes_per_param=16)\` → the smallest number of GPUs G such that state / G + activation_bytes ≤ gpu_bytes (each GPU also needs room for its activations). Return \`None\` if the activations alone don't fit.
- \`mfu(n_params, tokens_per_sec, n_gpus, peak_flops)\` → 6 · N · tokens_per_sec / (n_gpus · peak_flops)

\`\`\`text
min_gpus(70e9, 80e9, 20e9)  ->  19      (1.12e12 bytes of state, 60e9 free per GPU)
\`\`\``,
        starter: `
import math

def state_bytes(n_params, bytes_per_param=16):
    # TODO
    pass

def zero3_bytes_per_gpu(n_params, n_gpus, bytes_per_param=16):
    # TODO
    pass

def min_gpus(n_params, gpu_bytes, activation_bytes, bytes_per_param=16):
    # TODO
    pass

def mfu(n_params, tokens_per_sec, n_gpus, peak_flops):
    # TODO
    pass
`,
        solution: `
import math

def state_bytes(n_params, bytes_per_param=16):
    return n_params * bytes_per_param

def zero3_bytes_per_gpu(n_params, n_gpus, bytes_per_param=16):
    return n_params * bytes_per_param / n_gpus

def min_gpus(n_params, gpu_bytes, activation_bytes, bytes_per_param=16):
    free = gpu_bytes - activation_bytes
    if free <= 0:
        return None
    return max(1, math.ceil(n_params * bytes_per_param / free))

def mfu(n_params, tokens_per_sec, n_gpus, peak_flops):
    return 6 * n_params * tokens_per_sec / (n_gpus * peak_flops)
`,
        hints: ['Free memory per GPU = gpu_bytes − activation_bytes.', 'Round up with `math.ceil`, and never return fewer than 1 GPU.'],
        explain: `Back-of-the-envelope memory math is the first step of planning any training run: it tells you whether you need sharding, checkpointing or simply a smaller batch.`,
        tests: [
          { call: 'state_bytes(7e9)', expect: '1.12e11' },
          { call: 'min_gpus(70e9, 80e9, 20e9)', expect: '19' },
          { name: 'activations alone do not fit', call: 'min_gpus(1e9, 80e9, 90e9)', expect: 'None', hidden: true },
          { name: 'small model fits on one GPU', call: '[min_gpus(124e6, 24e9, 4e9), zero3_bytes_per_gpu(8e9, 32)]', expect: '[1, 4e9]', hidden: true },
          { name: 'MFU', call: 'mfu(8e9, 2.0e6, 256, 989e12)', expect: '6 * 8e9 * 2.0e6 / (256 * 989e12)' }
        ]
      },
      {
        id: 'grad-accumulation', title: 'Gradient Accumulation', difficulty: 'medium',
        prompt: `For linear regression with the MSE loss, implement:

- \`grad_mse(w, X, y)\` → (2/n) · Xᵀ (Xw − y), the full-batch gradient (n = len(y)).
- \`accumulated_grad(w, X, y, micro)\` → the same gradient computed micro-batch by micro-batch: split the rows into consecutive chunks of size \`micro\` (the last one may be smaller), compute each chunk's gradient with \`grad_mse\`, and add them up **weighted by chunk size / n**.
- \`effective_batch(micro, accum, gpus)\` → micro × accum × gpus.

The result must equal the full-batch gradient even when n is not a multiple of micro.`,
        starter: `
import numpy as np

def grad_mse(w, X, y):
    # TODO
    pass

def accumulated_grad(w, X, y, micro):
    # TODO
    pass

def effective_batch(micro, accum, gpus):
    # TODO
    pass
`,
        solution: `
import numpy as np

def grad_mse(w, X, y):
    X, y, w = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    return 2 / len(y) * X.T @ (X @ w - y)

def accumulated_grad(w, X, y, micro):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    g = np.zeros(X.shape[1])
    for start in range(0, n, micro):
        Xb, yb = X[start:start + micro], y[start:start + micro]
        g += grad_mse(w, Xb, yb) * len(yb) / n
    return g

def effective_batch(micro, accum, gpus):
    return micro * accum * gpus
`,
        hints: ['Each chunk\'s gradient is an average over its own rows.', 'Multiply by len(chunk) / n to turn averages back into the full-batch average.'],
        explain: `In frameworks, dividing each micro-batch loss by the number of accumulation steps works when all micro-batches have the same size. With a smaller last chunk, weighting by chunk size is what keeps the result exactly equal to the big-batch gradient.`,
        tests: [
          { name: 'full-batch gradient', call: 'grad_mse(np.array([1.0]), np.array([[1.0], [2.0]]), np.array([1.0, 1.0]))', expect: 'np.array([2.0])' },
          { name: 'accumulation equals the full gradient', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(10, 3)); y = rng.normal(size=10); w = rng.normal(size=3)', code: 'full = grad_mse(w, X, y)\nfor micro in (1, 2, 3, 4, 10, 16):\n    assert np.allclose(accumulated_grad(w, X, y, micro), full), micro' },
          { call: 'effective_batch(8, 16, 8)', expect: '1024' },
          { name: 'larger problem vs reference', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(37, 5)); y = rng.normal(size=37); w = np.zeros(5)', call: 'accumulated_grad(w, X, y, 8)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models', url: 'https://arxiv.org/abs/1910.02054' },
      { title: 'Shoeybi et al., Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism', url: 'https://arxiv.org/abs/1909.08053' },
      { title: 'Hugging Face, The Ultra-Scale Playbook: Training LLMs on GPU Clusters', url: 'https://huggingface.co/spaces/nanotron/ultrascale-playbook' },
      { title: 'PyTorch, Getting started with Fully Sharded Data Parallel', url: 'https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'reasoning-models', branch: 'llm', title: 'Reasoning Models & Test-Time Compute', short: 'Reasoning Models', level: 3, minutes: 40, prereqs: ['alignment', 'decoding'],
    tags: ['reasoning', 'chain of thought', 'test-time compute', 'self-consistency', 'majority vote', 'best-of-n', 'verifier', 'pass@k', 'rlvr', 'grpo', 'process reward model', 'thinking budget'],
    summary: 'Why letting a model think longer makes it smarter: chain-of-thought, sampling many answers with majority voting or a verifier (best-of-n), training reasoning with reinforcement learning on verifiable rewards (GRPO), process versus outcome rewards, thinking budgets, and the trade-offs in cost and latency.',
    why: `Recent jumps in LLM capability have come not only from bigger pretraining but from **test-time compute**: models that write out long reasoning before answering, trained with reinforcement learning on problems whose answers can be checked. Understanding how this works (and how to spend extra inference compute with voting and verifiers) explains modern "thinking" models and helps you decide when the extra cost and latency are worth paying.`,
    objectives: [
      'Explain why intermediate reasoning tokens improve multi-step problem solving',
      'Use self-consistency (majority voting) and best-of-n with a verifier, and estimate pass@k',
      'Describe reinforcement learning with verifiable rewards, GRPO\'s group-relative advantages, and process versus outcome rewards',
      'Decide when to use a reasoning model and how to set its thinking budget'
    ],
    sections: [
      { type: 'text', title: 'Thinking with tokens', md: `A transformer performs a fixed amount of computation for each token it produces. A hard problem (a long calculation, a proof, a tricky bug) needs more serial steps than one token's worth. Writing out intermediate steps (**chain of thought**) gives the model scratch space: every new token can build on the results written before it.

- 2022: *prompted* chain of thought ("let's think step by step") improved math and logic benchmarks.
- 2024 onwards: models **trained** to produce long internal reasoning (exploring approaches, checking intermediate results, backtracking) before giving a final answer. On math and coding benchmarks, accuracy keeps improving as the model is allowed to think longer.

This is called **test-time (inference-time) compute**: spending more computation per question at inference instead of only making the model bigger.` },
      { type: 'text', title: 'Spending compute at inference: voting and verifiers', md: `Even without special training you can trade compute for accuracy by **sampling several answers**:

- **Self-consistency (majority voting)**: sample k reasoning paths at a moderate temperature and return the most common final answer. If the right answer is more likely than any *single* wrong one, voting amplifies it.
- **Best-of-n with a verifier**: generate n candidates and let a checker pick (unit tests for code, a symbolic checker for math, or a learned reward model).

With a perfect verifier, you succeed whenever at least one of the n samples is right:

$$ P(at least one correct) = 1 − (1 − p)^n $$
:::where
p — the probability that a single sample is correct
n — the number of independent samples
(1 − p)^n — the probability that all n samples are wrong
:::

This is the idea behind the **pass@k** metric for code generation. Real verifiers are imperfect, and samples are not fully independent, so gains flatten out.` },
      { type: 'code', title: 'Self-consistency and best-of-n, simulated', code: `
import numpy as np
rng = np.random.default_rng(0)
p_correct, n_wrong, questions = 0.4, 5, 2000      # each sample: right answer (0) with p = 0.4, else one of 5 wrong ones

def sample_answers(k):
    right = rng.random((questions, k)) < p_correct
    wrong = rng.integers(1, n_wrong + 1, (questions, k))
    return np.where(right, 0, wrong)

for k in (1, 3, 5, 11, 21):
    answers = sample_answers(k)
    counts = np.stack([(answers == a).sum(axis=1) for a in range(n_wrong + 1)], axis=1)
    majority = np.argmax(counts + 0.1 * rng.random(counts.shape), axis=1)   # random tie-breaking
    vote_acc = np.mean(majority == 0)
    best_of_k = np.mean((answers == 0).any(axis=1))                          # a perfect verifier picks a right one
    print(f"k = {k:2d}   majority vote {vote_acc:.3f}   best-of-{k} with a verifier {best_of_k:.3f}   (1 - (1 - p)^k = {1 - (1 - p_correct) ** k:.3f})")
`, after: `A single sample is right 40% of the time. Because each wrong answer is less common than the right one, majority voting climbs well above 40%, and a verifier that can recognize a correct answer does even better.` },
      { type: 'chart', title: "Spending more samples: majority vote versus a verifier", md: `The simulation above, extended: each sample is right with probability 0.4, otherwise one of 5 wrong answers:`, after: `Both methods turn extra samples into accuracy, with diminishing returns on a log scale. Majority voting works because the right answer is more common than any single wrong one; it levels off near 100% here only because the wrong answers are spread out. With a verifier the curve climbs much faster, since one correct sample is enough, which is why reasoning systems invest in checkers: unit tests, proof checkers and trained verifiers.`, chart: {"kind":"line","x":[1,2,3,5,8,11,16,21,32,48,64],"xLog":true,"series":[{"name":"best-of-k with a perfect verifier (pass@k)","y":[0.407,0.625,0.774,0.921,0.98,0.997,1,1,1,1,1]},{"name":"majority vote over k samples","y":[0.407,0.396,0.456,0.584,0.667,0.751,0.843,0.906,0.961,0.989,0.998]},{"name":"a single sample","y":[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4],"dash":true}],"xLabel":"number of sampled answers k (log scale)","yLabel":"accuracy","yMin":0,"yMax":1} },
      { type: 'check', question: { type: 'fill', q: 'Each sample is correct with probability 0.5. With a perfect verifier, what is the chance that best-of-3 succeeds? (3 decimals)', answer: ['0.875', '.875', '7/8'], explain: '1 − 0.5³ = 0.875.' } },
      { type: 'text', title: 'Training reasoning with reinforcement learning', md: `**Reinforcement learning with verifiable rewards (RLVR)** trains on problems whose answers can be checked automatically: math with known results, code with unit tests, logic puzzles. For each prompt, sample a **group** of G responses, give each a reward (1 if the final answer is correct, often plus a small format reward), and increase the probability of the better-than-average responses.

**GRPO** (group relative policy optimization) uses the group itself as the baseline, no separate value network as in PPO:

$$ A_i = ( r_i − mean(r_1, …, r_G) ) / std(r_1, …, r_G) $$
:::where
r_i — the reward of response i in the group
G — the number of responses sampled for the same prompt
A_i — the advantage: how much better response i did than its siblings; every token of response i is pushed up (A_i > 0) or down (A_i < 0)
:::

A KL penalty to a reference model keeps the policy from drifting too far. With enough RL, models learn to reason at length, verify their own steps and backtrack, without being shown how.

- **Outcome rewards** check only the final answer: cheap and hard to game, but sparse.
- **Process reward models** score each intermediate step: denser feedback, but expensive to label and easier to exploit.
- Watch for **reward hacking**: code that special-cases the tests, answers formatted to fool the checker.` },
      { type: 'code', title: 'Group-relative advantages', code: `
import numpy as np
def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)

for rewards in ([1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]):
    print(rewards, "->", group_advantages(rewards).round(2))
`, after: `A rare success earns a large positive advantage. When every response in a group gets the same reward (all right or all wrong) every advantage is zero and the prompt teaches nothing, which is why training data must be neither too easy nor too hard for the current model.` },
      { type: 'text', title: "How far does thinking go?", md: `Reasoning models show a second kind of scaling: accuracy on hard problems rises roughly with the logarithm of the compute spent *at inference*, by thinking longer or sampling more. Doubling the thinking budget gives a steady gain, but each doubling buys less, and some problems stay out of reach at any budget the model can use well.

Limits worth knowing:
- **Overthinking**: on easy questions long reasoning adds cost and delay, and can even talk the model out of a correct first answer.
- **Faithfulness**: the written chain of thought is not guaranteed to be the computation that produced the answer. Models sometimes reach answers through shortcuts their reasoning text does not mention, so a plausible explanation is not proof of correct reasoning.
- **Verification is the bottleneck**: test-time compute pays off most where answers can be checked (tests, proofs, exact results); for open-ended questions, more samples mainly need a better judge.
- **Distillation**: reasoning traces from a large model can teach a small one to reason far better than its size suggests, which is how many small reasoning models are made.` },
      { type: 'text', title: 'Using reasoning models in practice', md: `- **Good fit**: math, code, data analysis, planning, multi-step agentic tasks, careful reading of long documents.
- **Poor fit**: simple lookups, classification, chit-chat, latency-critical steps (the extra thinking only adds cost and delay).
- **Control the budget**: most APIs expose an effort level or a thinking-token budget; tune it per task rather than globally.
- **Prompting changes**: give a clear goal, constraints and success criteria; avoid long step-by-step recipes, which can make a reasoning model worse.
- **Verify** with tools when you can: run the code, check the arithmetic, cite the source.
- **Distillation**: reasoning traces from large models are used to train smaller models to reason.
- **Caveat**: the written reasoning is a useful view of the model's process, but it is not guaranteed to be a complete or faithful account of how the answer was produced.` },
      { type: 'check', question: { type: 'tf', q: 'In GRPO, if all 8 responses to a prompt receive the same reward, that prompt produces no learning signal.', answer: true, explain: 'Every advantage is zero.' } }
    ],
    when: `- Problems with many dependent steps: math, coding, debugging, planning, analysis.
- When answers can be verified, add sampling + a verifier (tests, checkers) for extra accuracy.
- Train with RL on verifiable rewards when you have checkable tasks and the compute to sample many responses per prompt.
- Prefer a fast non-reasoning call for simple, high-volume or latency-sensitive steps.`,
    pitfalls: `- Paying for long reasoning on easy tasks.
- Majority voting over free-form text (normalize answers before voting).
- Verifiers that can be gamed (weak tests), leading to reward hacking.
- Training prompts that are all too easy or too hard (zero advantage everywhere).
- Treating the visible reasoning as a guaranteed faithful explanation.`,
    takeaways: [
      'More reasoning tokens = more computation per question; it helps multi-step problems.',
      'Self-consistency votes over samples; best-of-n uses a verifier: P(success) = 1 − (1 − p)^n.',
      'RL with verifiable rewards trains models to reason; GRPO uses group-relative advantages.',
      'Outcome rewards are sparse but robust; process rewards are dense but costly and gameable.',
      'Use reasoning effort where the task needs it and verify outputs with tools.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why can writing out reasoning steps improve an LLM\'s answers?', options: ['It makes the model bigger', 'Each generated token adds computation, and later tokens can build on earlier intermediate results', 'It changes the tokenizer', 'It reduces the temperature'], answer: 1, explain: 'The model gets scratch space for multi-step work.' },
      { type: 'mcq', q: 'When does majority voting improve accuracy over a single sample?', options: ['Always, by any amount', 'When the correct answer is more likely than any single wrong answer', 'Only at temperature 0', 'Never'], answer: 1, explain: 'Voting amplifies the most common answer.' },
      { type: 'fill', q: 'A group of 4 responses gets rewards [1, 0, 0, 0]. What is the group mean reward used as the GRPO baseline?', answer: ['0.25', '1/4'], explain: '(1 + 0 + 0 + 0) / 4 = 0.25.' },
      { type: 'tf', q: 'A process reward model scores intermediate reasoning steps, not just the final answer.', answer: true, explain: 'Denser signal, but harder to build and easier to game.' },
      { type: 'mcq', q: 'Which task benefits least from a long thinking budget?', options: ['Proving a geometry theorem', 'Debugging a race condition', 'Classifying a tweet as positive or negative', 'Planning a multi-step data migration'], answer: 2, explain: 'Simple classification gains little from long reasoning.' },
      {"type":"mcq","q":"Each sample is right with probability 0.4. Roughly how many samples does a perfect verifier need to succeed about 98% of the time?","options":["2","8","32","100"],"answer":1,"explain":"1 − 0.6⁸ ≈ 0.983: with 8 samples the chance that none is right drops below 2%."},
      {"type":"tf","q":"A model's written chain of thought always faithfully shows how it reached its answer.","answer":false,"explain":"Studies find cases where the stated reasoning omits what actually drove the answer, so reasoning text should be treated as evidence, not proof."}
    ],
    problems: [
      {
        id: 'majority-vote', title: 'Self-Consistency Voting', difficulty: 'easy',
        prompt: `Implement:

- \`majority_vote(answers)\` → the most common answer in the list; ties go to the answer that **appeared first**.
- \`vote_accuracy(samples, truths)\` → the fraction of questions whose majority vote equals the true answer (\`samples[i]\` is the list of sampled answers for question i).

\`\`\`text
majority_vote(["12", "15", "12", "9"])  ->  "12"
majority_vote(["a", "b", "b", "a"])     ->  "a"      (tie: "a" appeared first)
\`\`\``,
        starter: `
def majority_vote(answers):
    # TODO
    pass

def vote_accuracy(samples, truths):
    # TODO
    pass
`,
        solution: `
def majority_vote(answers):
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts.values())
    for a in answers:
        if counts[a] == best:
            return a

def vote_accuracy(samples, truths):
    hits = sum(majority_vote(s) == t for s, t in zip(samples, truths))
    return hits / len(truths)
`,
        hints: ['Count with a dict (insertion order follows first appearance).', 'Scan the answers in order and return the first one with the maximum count.'],
        explain: `Real implementations first normalize answers (extracting the final number, stripping units and whitespace) so that "12", "12.0" and "twelve" are counted as the same vote.`,
        tests: [
          { call: 'majority_vote(["12", "15", "12", "9"])', expect: '"12"' },
          { call: 'majority_vote(["a", "b", "b", "a"])', expect: '"a"' },
          { name: 'accuracy over questions', call: 'vote_accuracy([["1", "1", "2"], ["3", "4", "4"], ["7"]], ["1", "3", "7"])', expect: '2 / 3' },
          { name: 'numbers as answers', call: '[majority_vote([3, 1, 2, 2, 3, 1, 1]), majority_vote([5])]', expect: '[1, 5]', hidden: true }
        ]
      },
      {
        id: 'pass-at-k', title: 'Estimating pass@k', difficulty: 'medium',
        prompt: `You generated n samples for a problem and c of them are correct. The unbiased estimate of pass@k (the chance that at least one of k samples drawn from these n is correct) is

$$ pass@k = 1 − C(n − c, k) / C(n, k) $$
:::where
n — the number of samples generated
c — how many of them were correct
k — the number of attempts allowed
C(a, b) — "a choose b", the number of ways to pick b items from a (\`math.comb\`)
C(n − c, k) / C(n, k) — the probability that k samples drawn without replacement are all wrong
:::

Implement \`pass_at_k(n, c, k)\` (return 1.0 when n − c < k, because some draw must then be correct) and \`best_of_n(p, n)\` → 1 − (1 − p)^n for independent samples with success probability p.

\`\`\`text
pass_at_k(10, 3, 1)  ->  0.3
pass_at_k(5, 2, 2)   ->  1 − 3/10 = 0.7
\`\`\``,
        starter: `
import math

def pass_at_k(n, c, k):
    # TODO
    pass

def best_of_n(p, n):
    # TODO
    pass
`,
        solution: `
import math

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def best_of_n(p, n):
    return 1.0 - (1.0 - p) ** n
`,
        hints: ['`math.comb(a, b)` counts combinations.', 'If there are fewer wrong samples than k, every draw of k must include a correct one.'],
        explain: `Generating n > k samples and using this formula gives a much lower-variance estimate than literally drawing k samples once, which is why code benchmarks report pass@k this way.`,
        tests: [
          { call: '[pass_at_k(10, 3, 1), pass_at_k(5, 2, 2)]', expect: '[0.3, 0.7]' },
          { name: 'edge cases', call: '[pass_at_k(10, 0, 5), pass_at_k(10, 10, 1), pass_at_k(4, 2, 3)]', expect: '[0.0, 1.0, 1.0]' },
          { call: 'best_of_n(0.5, 3)', expect: '0.875' },
          { name: 'larger values vs reference', call: '[pass_at_k(200, 13, 10), pass_at_k(200, 13, 100), best_of_n(0.07, 20)]', hidden: true }
        ]
      },
      {
        id: 'grpo-advantages', title: 'GRPO Advantages', difficulty: 'medium',
        prompt: `Implement the core of group-relative policy optimization:

- \`group_advantages(rewards, eps=1e-8)\` → (r − mean(r)) / (std(r) + eps), using the population standard deviation (\`np.std\` default).
- \`informative_groups(groups)\` → the indices of the groups (lists of rewards) whose rewards are not all identical.
- \`pg_loss(logprobs, advantages)\` → −mean(advantage × logprob), where \`logprobs[i]\` is the summed log-probability of response i. Minimizing it raises the probability of responses with positive advantage.

\`\`\`text
group_advantages([1, 0, 0, 0])  ->  [1.732, -0.577, -0.577, -0.577]
\`\`\``,
        starter: `
import numpy as np

def group_advantages(rewards, eps=1e-8):
    # TODO
    pass

def informative_groups(groups):
    # TODO
    pass

def pg_loss(logprobs, advantages):
    # TODO
    pass
`,
        solution: `
import numpy as np

def group_advantages(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)

def informative_groups(groups):
    return [i for i, g in enumerate(groups) if len(set(g)) > 1]

def pg_loss(logprobs, advantages):
    return float(-np.mean(np.asarray(advantages, dtype=float) * np.asarray(logprobs, dtype=float)))
`,
        hints: ['`np.std` divides by n by default (population standard deviation).', 'A group is informative when `len(set(group)) > 1`.'],
        explain: `Normalizing within the group means only *relative* quality matters, so the same code works whatever the reward scale. Filtering out groups with identical rewards saves compute on prompts that teach nothing.`,
        tests: [
          { call: 'group_advantages([1, 0, 0, 0])', expect: 'np.array([3.0, -1.0, -1.0, -1.0]) / np.sqrt(3)' },
          { name: 'identical rewards give zero advantages', call: 'group_advantages([1, 1, 1])', expect: 'np.zeros(3)' },
          { call: 'informative_groups([[1, 1], [0, 1], [0, 0, 0], [0.5, 0.2]])', expect: '[1, 3]' },
          { name: 'policy-gradient loss', call: 'pg_loss([-10.0, -12.0], [1.0, -1.0])', expect: '-1.0' },
          { name: 'random rewards vs reference', setup: 'rng = np.random.default_rng(0)\nr = rng.integers(0, 2, 16).astype(float) + 0.1 * rng.random(16)\nlp = -rng.uniform(5, 50, 16)', call: '[group_advantages(r), pg_loss(lp, group_advantages(r))]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Wei et al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models', url: 'https://arxiv.org/abs/2201.11903' },
      { title: 'Wang et al., Self-Consistency Improves Chain of Thought Reasoning', url: 'https://arxiv.org/abs/2203.11171' },
      { title: 'Chen et al., Evaluating Large Language Models Trained on Code (pass@k)', url: 'https://arxiv.org/abs/2107.03374' },
      { title: 'Shao et al., DeepSeekMath (introduces GRPO)', url: 'https://arxiv.org/abs/2402.03300' }
    ]
  },

  // =====================================================================================
  {
    id: 'efficient-attention', branch: 'llm', title: 'Efficient Attention & Long Context', short: 'Long Context', level: 3, minutes: 40, prereqs: ['gpt', 'positional-encoding'],
    tags: ['flashattention', 'online softmax', 'kv cache', 'long context', 'sliding window attention', 'paged attention', 'context extension', 'rope scaling', 'state space models', 'linear attention'],
    summary: 'Why attention gets expensive and how modern models handle long inputs: the quadratic cost, FlashAttention\'s tiling with an online softmax, taming the KV cache (GQA, sliding windows, paging, prefix caching), extending the context by rescaling RoPE positions, and alternatives such as state-space models.',
    why: `Context windows grew from 2,048 tokens (GPT-3) to hundreds of thousands and beyond. That required attacking attention's **quadratic cost** and the **KV cache's memory** from every angle. These techniques explain why long prompts cost what they do, why some models handle long documents better than others, and how ideas from earlier lessons (RoPE, grouped-query attention, quantization) fit together.`,
    objectives: [
      'Quantify attention\'s compute and memory as a function of context length',
      'Explain FlashAttention: tiling plus an online softmax that never stores the full T × T matrix',
      'Compare techniques that shrink or bound the KV cache',
      'Describe context extension by position interpolation, and alternatives such as state-space models'
    ],
    sections: [
      { type: 'text', title: 'Where the cost comes from', md: `In every layer, attention compares every token with every other token. For context length T and model width d:

$$ attention FLOPs per layer ≈ 4 · T² · d          feed-forward FLOPs per layer ≈ 16 · T · d² $$
:::where
T — the number of tokens in the context
d — the model width (d_model)
4 · T² · d — the QKᵀ scores (2·T²·d) plus the weighted sum of values (2·T²·d)
16 · T · d² — the two feed-forward matrices of width 4d
:::

The attention share is T / (T + 4d): small for short prompts, dominant for long ones. Naive attention also **stores** the T × T score matrix: 16 billion numbers per head at 128k tokens. And during generation the **KV cache** grows linearly with T.` },
      { type: 'code', title: 'How the cost shifts with context length', code: `
d, layers = 4096, 32
for T in (2_048, 8_192, 32_768, 131_072):
    attn = layers * 4 * T * T * d
    ffn = layers * 16 * T * d * d
    print(f"T = {T:>7,}:  attention {attn:.2e} FLOPs   feed-forward {ffn:.2e} FLOPs   attention share {attn / (attn + ffn):.0%}")
` },
      { type: 'text', title: 'FlashAttention: same result, far less memory traffic', md: `GPUs are usually limited by **memory bandwidth**, not arithmetic. Naive attention writes the full score matrix to slow GPU memory (HBM) and reads it back for the softmax and the weighted sum. **FlashAttention** splits Q, K and V into tiles that fit in fast on-chip SRAM, processes one block of keys at a time, and combines the blocks with an **online softmax** that keeps a running maximum and a running sum:

$$ m′ = max( m, max_j s_j )          ℓ′ = ℓ · e^(m − m′) + Σ_j e^(s_j − m′)          o′ = o · e^(m − m′) + Σ_j e^(s_j − m′) v_j $$
:::where
s_j — the scores of the keys in the new block
m — the running maximum score seen so far (m′ is the updated value)
ℓ — the running sum of exponentials, measured relative to m
o — the running weighted sum of value vectors v_j
e^(m − m′) — a correction factor that rescales the old sums when a larger score appears
The final output is o / ℓ.
:::

The result is **exactly** standard attention, with memory that grows linearly in T and typically 2–4× higher speed. It is now the default in PyTorch (\`scaled_dot_product_attention\`) and every serious training stack.` },
      { type: 'chart', title: "Extra memory for attention in one layer", md: `One layer with 32 heads of width 128, for a single sequence:`, after: `Writing out the full score matrix needs T² numbers per head: about 17 GB per layer at 16k tokens and over a terabyte at 128k, impossible on any GPU. FlashAttention never stores it: it streams tiles of keys and values through fast on-chip memory and keeps only running softmax statistics, so its memory grows linearly with T. The arithmetic is the same; what changes is how much data moves.`, chart: {"kind":"line","x":[1024,4096,16384,65536,131072],"xLog":true,"yLog":true,"series":[{"name":"materialise the T × T score matrix (32 heads)","y":[0.06711,1.074,17.18,274.9,1100]},{"name":"FlashAttention: Q, K, V and output only","y":[0.02517,0.1007,0.4027,1.611,3.221]}],"xLabel":"context length T (log scale)","yLabel":"GB per layer, 16-bit (log scale)"} },
      { type: 'code', title: 'Online softmax: attention one block at a time', code: `
import numpy as np
rng = np.random.default_rng(0)
T, d, block = 1000, 16, 128
q = rng.normal(size=d)
K, V = rng.normal(size=(T, d)), rng.normal(size=(T, d))

s = K @ q / np.sqrt(d)                      # standard attention for one query: all T scores at once
w = np.exp(s - s.max())
full = (w / w.sum()) @ V

m, l, o = -np.inf, 0.0, np.zeros(d)         # FlashAttention-style: running max, running sum, running output
for start in range(0, T, block):
    sb = K[start:start + block] @ q / np.sqrt(d)
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new)               # correct what was accumulated with the old maximum
    p = np.exp(sb - m_new)
    l = l * scale + p.sum()
    o = o * scale + p @ V[start:start + block]
    m = m_new
print("largest difference from standard attention:", float(np.abs(full - o / l).max()))
print(f"scores held in memory at once: {block} instead of {T}")
` },
      { type: 'check', question: { type: 'tf', q: 'FlashAttention computes an approximation of attention to save memory.', answer: false, explain: 'It computes exactly the same result; it only reorganizes the computation to avoid storing the full score matrix.' } },
      { type: 'text', title: 'Taming the KV cache', md: `Each generated token adds keys and values for every layer to the cache (see *Quantization & Efficient Inference*). Ways to keep it manageable:

- **Fewer KV heads**: grouped-query or multi-query attention share keys and values across query heads.
- **KV-cache quantization** to 8 or 4 bits.
- **Sliding-window attention**: each token attends only to the previous W tokens, so the cache never exceeds W entries; stacking layers still lets information travel further.
- **Attention sinks**: keep the first few tokens plus a recent window, which stabilizes streaming generation.
- **PagedAttention** (vLLM): store the cache in fixed-size pages like virtual memory (no fragmentation, and requests can share pages).
- **Prefix caching**: reuse the cache of a repeated prefix such as a long system prompt across requests.
- **Latent attention**: compress keys and values into a small low-rank latent vector (DeepSeek's multi-head latent attention).` },
      { type: 'text', title: 'Extending the context window', md: `A model trained on 8k-token sequences fails beyond that length: rotary embeddings produce angles it has never seen. **Position interpolation** squeezes the new, longer range of positions into the trained range:

$$ pos′ = pos · L_train / L_new $$
:::where
pos — the real position of a token in the long input
L_train — the context length used in training
L_new — the longer target context length
pos′ — the rescaled position fed into RoPE, which stays within the trained range
:::

Refinements such as NTK-aware scaling and YaRN rescale different RoPE frequencies by different amounts, preserving fine local detail. A short fine-tune on long documents then adapts the model. Test long-context ability with realistic tasks, not only "needle in a haystack" retrieval: models often use the middle of long contexts less reliably than the beginning and end.` },
      { type: 'text', title: "Local, global and sparse attention patterns", md: `Not every token needs to see every other token. Restricting which keys each query attends to cuts both the compute and the KV cache:

| Pattern | Each token attends to | Used in |
| --- | --- | --- |
| sliding window (local) | the previous W tokens | Mistral 7B, and the local layers of many recent models |
| global tokens | a few special tokens that attend to, and are attended by, everything | Longformer, BigBird |
| interleaved local and global layers | a window in most layers, the full context every few layers | Gemma 2 and 3, several recent long-context models |
| attention sinks | the first few tokens plus a recent window | streaming generation without a growing cache |

Stacked windowed layers still let information travel far, just as stacked convolutions grow their receptive field: after L layers with window W, a token can be influenced by roughly L × W earlier tokens. Interleaving a few full-attention layers restores direct long-range lookups, which pure windows handle poorly, while keeping most layers cheap.` },
      { type: 'text', title: 'Beyond softmax attention', md: `- **Linear attention** replaces the softmax with kernel feature maps so the cost becomes linear in T, at some cost in quality.
- **State-space models** (S4, Mamba) process sequences like a recurrent network with a cleverly parameterized state: linear time and constant memory per generated token.
- **Hybrids** combine a few attention layers with many state-space or sliding-window layers.

The trade-off: full attention can retrieve any earlier token exactly; compressed-state models must decide what to remember, which can hurt precise recall over very long inputs.` },
      { type: 'check', question: { type: 'mcq', q: 'With sliding-window attention of window W, how large can the KV cache per layer get during generation?', options: ['It grows forever', 'At most W tokens', 'Exactly 1 token', 'T² entries'], answer: 1, explain: 'Older tokens are never attended to again, so they can be dropped.' } }
    ],
    when: `- Long documents, codebases, long conversations and agent transcripts.
- FlashAttention-style kernels: always, for training and inference.
- Sliding windows, paging and prefix caching for serving many long requests.
- Context extension when an existing model must handle longer inputs than it was trained on.`,
    pitfalls: `- Assuming a model uses its whole advertised context equally well: test on realistic long tasks.
- Sending enormous prompts when retrieval (RAG) would be cheaper and more accurate.
- Applying RoPE scaling without a short long-context fine-tune.
- Forgetting that cost and latency grow quickly with prompt length.
- Comparing long-context benchmarks that only test simple retrieval.`,
    takeaways: [
      'Attention costs ≈ 4·T²·d FLOPs per layer and dominates for long contexts.',
      'FlashAttention computes exact attention in tiles with an online softmax: linear memory, much faster.',
      'The KV cache grows with T; GQA, quantization, sliding windows, paging and prefix caching tame it.',
      'Context extension rescales RoPE positions (interpolation, NTK, YaRN) plus a short fine-tune.',
      'State-space and linear-attention models trade exact recall for linear cost.'
    ],
    quiz: [
      { type: 'mcq', q: 'Doubling the context length roughly does what to attention\'s compute?', options: ['Doubles it', 'Quadruples it', 'Leaves it unchanged', 'Halves it'], answer: 1, explain: 'The cost is proportional to T².' },
      { type: 'mcq', q: 'What does the online softmax keep while streaming over blocks of keys?', options: ['The full score matrix', 'A running maximum, a running sum of exponentials and a running weighted output', 'Only the last block', 'The gradients'], answer: 1, explain: 'Rescaling by e^(m − m′) keeps the result exact.' },
      { type: 'fill', q: 'A model trained on 4,096 tokens is extended to 16,384 with position interpolation. By what factor are positions multiplied?', answer: ['0.25', '1/4'], explain: 'L_train / L_new = 4096 / 16384 = 0.25.' },
      { type: 'tf', q: 'Prefix caching lets many requests that share the same long system prompt reuse its KV cache.', answer: true, explain: 'The shared prefix is computed once.' },
      { type: 'mcq', q: 'What is a known weakness of compressed-state models (e.g. state-space models) compared with full attention?', options: ['They cannot generate text', 'Exact recall of arbitrary earlier tokens over very long inputs', 'They need a causal mask', 'They are quadratic in length'], answer: 1, explain: 'A fixed-size state must decide what to forget.' },
      {"type":"fill","q":"A model has 24 layers, all with sliding-window attention over 4,096 tokens. Roughly how far back can information reach, in tokens?","answer":["98304","98,304","98k","about 98000"],"explain":"Information can hop one window per layer: about 24 × 4,096 = 98,304 tokens."},
      {"type":"mcq","q":"Why does materialising the full attention score matrix become impossible for long contexts?","options":["The softmax becomes unstable","Its size grows with the square of the context length","It needs more heads","The keys and values get longer"],"answer":1,"explain":"T tokens give T² scores per head; at 128k tokens that is over a terabyte per layer in 16-bit numbers."}
    ],
    problems: [
      {
        id: 'online-softmax', title: 'Attention with an Online Softmax', difficulty: 'medium',
        prompt: `Implement \`online_attention(q, K, V, block)\` → the attention output for one query (length d), processing the keys and values in consecutive blocks of \`block\` rows and never computing the softmax over all scores at once.

For each block: scores s = K_block @ q / √d; m′ = max(m, max(s)); rescale the running sum ℓ and running output o by e^(m − m′); add Σ e^(s − m′) to ℓ and Σ e^(s − m′) v to o. Start with m = −∞, ℓ = 0, o = 0 and return o / ℓ.

The result must match standard softmax attention, even with very large scores.`,
        starter: `
import numpy as np

def online_attention(q, K, V, block):
    q, K, V = (np.asarray(a, dtype=float) for a in (q, K, V))
    # TODO
    pass
`,
        solution: `
import numpy as np

def online_attention(q, K, V, block):
    q, K, V = (np.asarray(a, dtype=float) for a in (q, K, V))
    d = q.shape[0]
    m, l, o = -np.inf, 0.0, np.zeros(V.shape[1])
    for start in range(0, len(K), block):
        s = K[start:start + block] @ q / np.sqrt(d)
        m_new = max(m, float(s.max()))
        scale = np.exp(m - m_new)
        p = np.exp(s - m_new)
        l = l * scale + p.sum()
        o = o * scale + p @ V[start:start + block]
        m = m_new
    return o / l
`,
        hints: ['`np.exp(-np.inf - m_new)` is 0, so the first block needs no special case.', 'Rescale both ℓ and o whenever the maximum grows.'],
        explain: `The same two running statistics let attention be computed in tiles inside fast on-chip memory. FlashAttention applies this idea to many queries at once and fuses everything into one GPU kernel.`,
        tests: [
          { name: 'matches standard attention', setup: 'rng = np.random.default_rng(0)\nq = rng.normal(size=8); K = rng.normal(size=(50, 8)); V = rng.normal(size=(50, 3))\ns = K @ q / np.sqrt(8); w = np.exp(s - s.max()); w /= w.sum()', call: 'online_attention(q, K, V, 7)', expect: 'w @ V' },
          { name: 'block sizes do not matter', setup: 'rng = np.random.default_rng(1)\nq = rng.normal(size=4); K = rng.normal(size=(23, 4)); V = rng.normal(size=(23, 5))', code: 'ref = online_attention(q, K, V, 23)\nfor b in (1, 2, 5, 10, 100):\n    assert np.allclose(online_attention(q, K, V, b), ref), b' },
          { name: 'stable with huge scores', setup: 'rng = np.random.default_rng(2)\nq = 300 * rng.normal(size=6); K = rng.normal(size=(40, 6)); V = rng.normal(size=(40, 2))', code: 'out = online_attention(q, K, V, 8)\nassert np.all(np.isfinite(out))\ns = K @ q / np.sqrt(6)\nw = np.exp(s - s.max()); w /= w.sum()\nassert np.allclose(out, w @ V)', hidden: true },
          { name: 'random case vs reference', setup: 'rng = np.random.default_rng(3)\nq = rng.normal(size=5); K = rng.normal(size=(17, 5)); V = rng.normal(size=(17, 4))', call: 'online_attention(q, K, V, 4)', hidden: true }
        ]
      },
      {
        id: 'sliding-window-mask', title: 'Sliding-Window Attention', difficulty: 'easy',
        prompt: `Implement:

- \`sliding_window_mask(T, W)\` → a boolean (T, T) array where \`mask[i, j]\` is True when token i may attend to token j: j ≤ i (causal) and i − j < W (within the window).
- \`kv_cache_len(t, W)\` → how many past tokens must be kept in the cache when generating at position t (0-based) with window W: min(t + 1, W).
- \`attended_pairs(T, W)\` → the number of True entries in the mask.

\`\`\`text
sliding_window_mask(4, 2) -> [[T, F, F, F],
                              [T, T, F, F],
                              [F, T, T, F],
                              [F, F, T, T]]
\`\`\``,
        starter: `
import numpy as np

def sliding_window_mask(T, W):
    # TODO
    pass

def kv_cache_len(t, W):
    # TODO
    pass

def attended_pairs(T, W):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sliding_window_mask(T, W):
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    return (j <= i) & (i - j < W)

def kv_cache_len(t, W):
    return min(t + 1, W)

def attended_pairs(T, W):
    return int(sliding_window_mask(T, W).sum())
`,
        hints: ['Build index grids with `np.arange(T)[:, None]` and `np.arange(T)[None, :]`.', 'Combine the two conditions with `&`.'],
        explain: `With a window, each row has at most W True entries, so the total work is about T·W instead of T²/2: linear in the sequence length. Information still flows further than W through stacked layers.`,
        tests: [
          { call: 'sliding_window_mask(4, 2)', expect: 'np.array([[True, False, False, False], [True, True, False, False], [False, True, True, False], [False, False, True, True]])' },
          { call: '[kv_cache_len(0, 4096), kv_cache_len(10000, 4096)]', expect: '[1, 4096]' },
          { name: 'a full window is the causal mask', call: 'attended_pairs(6, 6)', expect: '21' },
          { name: 'pair counts', call: '[attended_pairs(10, 3), attended_pairs(100, 10)]', expect: '[27, 955]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention', url: 'https://arxiv.org/abs/2205.14135' },
      { title: 'Chen et al., Extending Context Window of LLMs via Positional Interpolation', url: 'https://arxiv.org/abs/2306.15595' },
      { title: 'Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention', url: 'https://arxiv.org/abs/2309.06180' },
      { title: 'Gu & Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces', url: 'https://arxiv.org/abs/2312.00752' }
    ]
  },

  // =====================================================================================
  {
    id: 'mixture-of-experts', branch: 'llm', title: 'Mixture of Experts (MoE)', short: 'Mixture of Experts', level: 3, minutes: 35, prereqs: ['transformer-block', 'pretraining'],
    tags: ['mixture of experts', 'moe', 'router', 'top-k gating', 'sparse model', 'load balancing', 'capacity factor', 'expert parallelism', 'active parameters'],
    summary: 'How models grow to hundreds of billions of parameters while spending the compute of a much smaller model: replacing the feed-forward layer with many expert networks and a router that sends each token to only a few, plus load balancing, capacity limits and the memory and serving trade-offs.',
    why: `Scaling laws say bigger models are better, but every parameter used for a token costs compute. **Mixture of Experts** decouples the two: total parameters (the model's capacity to store knowledge) grow with the number of experts, while the compute per token depends only on the few experts each token visits. Many frontier and open models use MoE, so "total versus active parameters" is essential vocabulary for reading model cards and planning hardware.`,
    objectives: [
      'Describe an MoE layer: experts, router, top-k gating and the weighted combination',
      'Distinguish total from active parameters and compute their compute and memory implications',
      'Explain load-balancing losses, capacity factors and token dropping',
      'Weigh MoE\'s benefits against its memory, communication and training challenges'
    ],
    sections: [
      { type: 'text', title: 'Experts and a router', md: `In an MoE transformer, the feed-forward network of (some) blocks is replaced by **E expert** feed-forward networks plus a small **router**: a single linear layer that scores the experts for each token. Only the top-k experts run for that token, and their outputs are mixed with gate weights:

$$ g = softmax( top_k( x W_r ) )          y = Σ_{e ∈ top-k} g_e · FFN_e(x) $$
:::where
x — one token's vector (length d)
W_r — the router's weight matrix (d × E), giving one score per expert
top_k — keep the k highest scores and discard the rest (k is often 1 or 2; some models pick 8 of 256)
g_e — the gate weight of chosen expert e (the k gates sum to 1)
FFN_e — expert e's own feed-forward network
y — the layer's output for this token
:::

Different tokens choose different experts. In practice experts specialize less by human topic and more by token types and syntax. Many recent designs also add one or two **shared experts** that every token uses.` },
      { type: 'demo', title: 'Diagram: routing one token', demo: 'dg-moe' },
      { type: 'text', title: 'Total versus active parameters', md: `| Model | Total parameters | Active per token | Experts |
| --- | --- | --- | --- |
| Mixtral 8x7B | ≈ 47B | ≈ 13B | 2 of 8 per layer |
| DeepSeek-V3 | ≈ 671B | ≈ 37B | 8 of 256 routed + 1 shared |

The attention layers and embeddings are shared, so the active count is:

$$ active ≈ shared + k · P_expert          total ≈ shared + E · P_expert $$
:::where
shared — parameters every token uses (attention, embeddings, norms, shared experts)
P_expert — the parameters of one routed expert (summed over all MoE layers)
k — the number of experts each token is routed to
E — the total number of routed experts
:::

**Compute** per token ≈ 2 × active parameters, like a dense model of that size. **Memory**, however, must hold *all* parameters: a 47B MoE needs the memory of a 47B dense model while running at roughly the speed of a 13B one.` },
      { type: 'chart', title: "Parameters stored versus parameters used per token", after: `A dense model uses every parameter for every token. Mixtral stores 47 billion parameters but runs only about 13 billion per token, and DeepSeek-V3 stores 671 billion but runs 37 billion: about as much compute per token as a mid-sized dense model, with the knowledge capacity of a much larger one. The catch is memory: all experts must be loaded, because any token may need any of them.`, chart: {"kind":"bar","x":["dense 70B","Mixtral 8x7B","DeepSeek-V3"],"yLog":true,"series":[{"name":"total (must be in memory)","y":[70,47,671]},{"name":"active per token (sets the compute)","y":[70,13,37]}],"xLabel":"model","yLabel":"billions of parameters (log scale)","yMin":1} },
      { type: 'code', title: 'An MoE layer in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
n_tokens, d, d_ff, E, k = 12, 8, 32, 4, 2
x = rng.normal(size=(n_tokens, d))
W_router = rng.normal(0, 0.5, (d, E))
experts = [(rng.normal(0, 0.3, (d, d_ff)), rng.normal(0, 0.3, (d_ff, d))) for _ in range(E)]

scores = x @ W_router                                         # (tokens, E)
top = np.argsort(-scores, axis=1)[:, :k]                      # the k best experts for each token
top_scores = np.take_along_axis(scores, top, axis=1)
gates = np.exp(top_scores - top_scores.max(axis=1, keepdims=True))
gates /= gates.sum(axis=1, keepdims=True)

out = np.zeros_like(x)
for t in range(n_tokens):
    for j in range(k):
        W1, W2 = experts[top[t, j]]
        out[t] += gates[t, j] * (np.maximum(0, x[t] @ W1) @ W2)   # weighted sum of the chosen experts

print("experts chosen by the first 5 tokens:", top[:5].tolist())
print("tokens routed to each expert:", [int(np.sum(top == e)) for e in range(E)])
p_expert = d * d_ff + d_ff * d
print(f"expert parameters: total {E * p_expert:,}, active per token {k * p_expert:,}")
` },
      { type: 'check', question: { type: 'fill', q: 'An MoE layer has 16 experts of 50M parameters each and routes every token to 2. How many expert parameters (in millions) are active per token?', answer: ['100', '100m'], explain: '2 × 50M = 100M (of 800M in total).' } },
      { type: 'text', title: 'Load balancing', md: `Left alone, routers **collapse**: a few experts get most tokens, train faster, look even better, and attract still more tokens, while the others sit idle. The Switch Transformer added an auxiliary **load-balancing loss**:

$$ L_balance = E · Σ_e f_e · P_e $$
:::where
E — the number of experts
f_e — the fraction of tokens in the batch whose top choice is expert e
P_e — the average router probability assigned to expert e
The loss reaches its minimum of 1 when both are uniform (1/E for every expert).
:::

Experts also have a **capacity**: each can process at most C × (tokens × k / E) tokens per batch (C is the capacity factor, e.g. 1.25). Overflowing tokens are dropped from that layer and simply pass through the residual connection. Newer models balance load without an auxiliary loss, for example by adjusting a per-expert bias in the router (DeepSeek-V3).` },
      { type: 'code', title: "Measuring load balance", code: `import numpy as np
rng = np.random.default_rng(0)
E, tokens = 8, 10_000

def balance_loss(router_probs):
    chosen = router_probs.argmax(axis=1)                       # top-1 routing, for simplicity
    f = np.bincount(chosen, minlength=E) / len(chosen)         # share of tokens sent to each expert
    P = router_probs.mean(axis=0)                              # average router probability per expert
    return E * np.sum(f * P), f

logits_ok = rng.normal(0, 1, (tokens, E))                      # a healthy router: no favourites
logits_bad = logits_ok + np.array([3.0, 2.0, 0, 0, 0, 0, 0, 0])  # a router that has started to favour two experts
losses = {}
for name, logits in [("healthy", logits_ok), ("collapsing", logits_bad)]:
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    losses[name], f = balance_loss(p)
    print(f"{name:<10} token shares {np.round(f, 2)}   balance loss {losses[name]:.2f}")
print("the loss rises as routing collapses:", losses["collapsing"] > 2 * losses["healthy"])`, after: `The healthy router spreads tokens almost exactly evenly (about 12.5% each) and the loss is 1.0, its minimum. Once two experts are favoured, they receive 97% of the tokens between them, six experts sit almost idle, and the loss rises to about 3.5. Adding this term to the training loss pushes the router back towards balance before the idle experts are wasted.`, predict: {"type":"mcq","q":"A router sends most tokens to two of its 8 experts. Compared with a balanced router (loss about 1), the balance loss will be…","options":["Lower","About the same","Clearly higher","Exactly zero"],"answer":2,"explain":"The loss multiplies each expert's token share by its average router probability; concentrating both on the same experts makes the sum, and the loss, large.","expect":["the loss rises as routing collapses: True"]} },
      { type: 'text', title: 'Trade-offs', md: `**Advantages**
- More parameters (knowledge) for the same compute per token; reaches a given loss with less training compute than a dense model.
- Fast inference relative to total size.

**Challenges**
- All experts must be in memory: MoE needs as much GPU memory as a dense model of its total size.
- **Expert parallelism** spreads experts over GPUs and needs all-to-all communication to send tokens to their experts and back.
- Training instabilities (router collapse, loss spikes) and trickier fine-tuning.
- Small inference batches leave many experts idle, which reduces hardware efficiency.

For local use the rule of thumb: an MoE needs the *memory* of its total size but runs at the *speed* of its active size.` },
      { type: 'check', question: { type: 'tf', q: 'An MoE model with 47B total and 13B active parameters needs about as much memory as a 13B dense model.', answer: false, explain: 'All 47B parameters must be stored; only the compute per token is like a 13B model.' } }
    ],
    when: `- Training the largest models efficiently: more capacity per FLOP.
- Serving at scale on clusters with plenty of memory and fast interconnects.
- Less attractive for memory-constrained local inference or tiny batch sizes.
- Dense models remain simpler to train, fine-tune and deploy at small and medium scale.`,
    pitfalls: `- Judging a model's cost by active parameters alone: memory follows total parameters.
- Router collapse when load balancing is missing or too weak.
- Capacity factors set too low (many dropped tokens) or too high (wasted compute).
- Fine-tuning that destabilizes the router; many recipes freeze or regularize it.`,
    takeaways: [
      'MoE replaces the FFN with E experts; a router sends each token to the top-k.',
      'Output = gate-weighted sum of the chosen experts\' outputs.',
      'Compute follows active parameters; memory follows total parameters.',
      'Load-balancing losses (E·Σ f_e·P_e) or router biases prevent expert collapse.',
      'Capacity factors bound each expert\'s work; overflow tokens skip the layer.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the router in an MoE layer do?', options: ['Splits the vocabulary between experts', 'Scores the experts for each token and selects the top-k to run', 'Averages all experts equally', 'Decides the learning rate'], answer: 1, explain: 'It is a small linear layer followed by top-k selection.' },
      { type: 'mcq', q: 'Why is a load-balancing loss needed?', options: ['To speed up the softmax', 'Without it, routing tends to collapse onto a few experts', 'To reduce memory', 'To make experts identical'], answer: 1, explain: 'Popular experts improve and attract even more tokens.' },
      { type: 'fill', q: 'With a perfectly uniform router over E experts, what value does the load-balancing loss E·Σ f_e·P_e take?', answer: ['1', '1.0'], explain: 'E · E · (1/E)(1/E) = 1.' },
      { type: 'tf', q: 'In an MoE transformer, attention layers are usually shared by all tokens while the feed-forward layers are split into experts.', answer: true, explain: 'The experts replace the FFNs.' },
      { type: 'mcq', q: 'What happens to a token that exceeds its expert\'s capacity?', options: ['The batch fails', 'It is dropped from that expert and passes through the residual connection', 'It is sent to every expert', 'It is deleted from the sequence'], answer: 1, explain: 'The residual path carries it to the next layer unchanged.' },
      {"type":"mcq","q":"DeepSeek-V3 has 671B total and 37B active parameters. What does its compute per token resemble?","options":["A 671B dense model","A dense model of roughly 37B parameters","A 7B model","It has no fixed compute"],"answer":1,"explain":"Compute per token depends on the active parameters, so it is comparable to a dense model of about 37B; memory, however, must hold all 671B."},
      {"type":"tf","q":"In an MoE model, every expert must be held in memory even though each token uses only a few of them.","answer":true,"explain":"Any token may be routed to any expert, so all experts have to be loaded (or quickly reachable) at all times."}
    ],
    problems: [
      {
        id: 'topk-router', title: 'Top-k Routing', difficulty: 'medium',
        prompt: `Implement an MoE layer's routing:

- \`route(x, W_r, k)\` → \`(idx, gates)\`: scores = x @ W_r has shape (n, E). For each token pick the k highest-scoring experts in descending order (\`np.argsort(-scores, axis=1, kind="stable")[:, :k]\`, so ties prefer the lower index); gates = softmax over those k scores (stable). Both outputs have shape (n, k).
- \`moe_forward(x, W_r, experts, k)\` → (n, d): \`experts\` is a list of functions; token t's output is Σ_j gates[t, j] · experts[idx[t, j]](x[t]).

\`\`\`text
x = [[1, 0]], W_r = [[3, 1, 2], [0, 0, 0]], k = 2  ->  idx = [[0, 2]], gates = softmax([3, 2])
\`\`\``,
        starter: `
import numpy as np

def route(x, W_r, k):
    # TODO
    pass

def moe_forward(x, W_r, experts, k):
    # TODO
    pass
`,
        solution: `
import numpy as np

def route(x, W_r, k):
    scores = np.asarray(x, dtype=float) @ np.asarray(W_r, dtype=float)
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(scores, idx, axis=1)
    top = top - top.max(axis=1, keepdims=True)
    gates = np.exp(top)
    gates = gates / gates.sum(axis=1, keepdims=True)
    return idx, gates

def moe_forward(x, W_r, experts, k):
    x = np.asarray(x, dtype=float)
    idx, gates = route(x, W_r, k)
    out = np.zeros_like(x)
    for t in range(len(x)):
        for j in range(k):
            out[t] += gates[t, j] * experts[idx[t, j]](x[t])
    return out
`,
        hints: ['`np.take_along_axis(scores, idx, axis=1)` gathers the chosen scores.', 'The softmax runs over the k selected scores only.', 'Loop over tokens and their k experts, accumulating the weighted outputs.'],
        explain: `Real implementations avoid the Python loop: they sort tokens by expert, run each expert once on its whole group of tokens (a batched matrix multiply), and scatter the results back, which is where all-to-all communication comes in when experts live on different GPUs.`,
        tests: [
          { call: 'route(np.array([[1.0, 0.0]]), np.array([[3.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), 2)', expect: '(np.array([[0, 2]]), np.exp([[3.0, 2.0]]) / np.exp([[3.0, 2.0]]).sum())' },
          { name: 'k = 1 picks one expert with gate 1', setup: 'experts = [lambda v: v * 0, lambda v: v * 10, lambda v: v + 1]\nW_r = np.array([[0.0, 5.0, 1.0], [0.0, 0.0, 9.0]])', call: 'moe_forward(np.array([[1.0, 0.0], [0.0, 1.0]]), W_r, experts, 1)', expect: 'np.array([[10.0, 0.0], [1.0, 2.0]])' },
          { name: 'random layer vs reference', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=(6, 4)); W_r = rng.normal(size=(4, 5))\nmats = [rng.normal(size=(4, 4)) for _ in range(5)]\nexperts = [lambda v, M=M: np.tanh(v @ M) for M in mats]', call: 'moe_forward(x, W_r, experts, 2)', hidden: true },
          { name: 'gates sum to one', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(size=(20, 3)); W_r = 10 * rng.normal(size=(3, 8))', code: 'idx, g = route(x, W_r, 3)\nassert idx.shape == (20, 3) and np.allclose(g.sum(axis=1), 1)\nscores = x @ W_r\nassert np.all(np.take_along_axis(scores, idx, axis=1)[:, 0] == scores.max(axis=1))', hidden: true }
        ]
      },
      {
        id: 'moe-balance', title: 'Load Balance and Parameter Counts', difficulty: 'easy',
        prompt: `Implement:

- \`load_balance_loss(probs)\` → E · Σ_e f_e · P_e, where \`probs\` is the (n, E) matrix of router probabilities (each row sums to 1), f_e is the fraction of tokens whose highest-probability expert is e, and P_e is the mean of column e.
- \`moe_params(shared, per_expert, n_experts, k)\` → \`(total, active)\` = (shared + n_experts · per_expert, shared + k · per_expert).

\`\`\`text
A perfectly balanced router gives a loss of 1.0; one that sends every token to the same expert with certainty gives E.
\`\`\``,
        starter: `
import numpy as np

def load_balance_loss(probs):
    # TODO
    pass

def moe_params(shared, per_expert, n_experts, k):
    # TODO
    pass
`,
        solution: `
import numpy as np

def load_balance_loss(probs):
    probs = np.asarray(probs, dtype=float)
    n, E = probs.shape
    top1 = probs.argmax(axis=1)
    f = np.array([np.mean(top1 == e) for e in range(E)])
    P = probs.mean(axis=0)
    return float(E * np.sum(f * P))

def moe_params(shared, per_expert, n_experts, k):
    return shared + n_experts * per_expert, shared + k * per_expert
`,
        hints: ['`probs.argmax(axis=1)` gives each token\'s top expert.', 'f_e: the mean of `top1 == e`; P_e: the column mean.'],
        explain: `The loss multiplies "how many tokens go to e" by "how much the router wants e"; its gradient (through P_e) nudges the router away from overloaded experts.`,
        tests: [
          { name: 'balanced routing', call: 'load_balance_loss(np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]))', expect: '1.0' },
          { name: 'collapsed routing', call: 'load_balance_loss(np.tile([1.0, 0.0, 0.0], (5, 1)))', expect: '3.0' },
          { call: 'moe_params(5e9, 1.4e9, 8, 2)', expect: '(5e9 + 8 * 1.4e9, 5e9 + 2 * 1.4e9)' },
          { name: 'random router vs reference', setup: 'rng = np.random.default_rng(0)\nz = rng.normal(size=(50, 6)); probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)', call: 'load_balance_loss(probs)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', url: 'https://arxiv.org/abs/1701.06538' },
      { title: 'Fedus et al., Switch Transformers', url: 'https://arxiv.org/abs/2101.03961' },
      { title: 'Jiang et al., Mixtral of Experts', url: 'https://arxiv.org/abs/2401.04088' },
      { title: 'DeepSeek-AI, DeepSeek-V3 Technical Report', url: 'https://arxiv.org/abs/2412.19437' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
