/* ML Atlas – content: Transformers & LLMs (core architecture) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'text-representation', branch: 'llm', title: 'Text as Data: Tokens, Bag-of-Words & TF-IDF', short: 'Text as Data', level: 1, minutes: 35, prereqs: ['feature-engineering', 'python-essentials'],
    tags: ['nlp', 'bag of words', 'tf-idf', 'n-grams', 'vocabulary', 'document similarity'],
    summary: 'The first step of NLP: turning text into numbers. Tokenizing, vocabularies, bag-of-words counts, n-grams, TF-IDF weighting — and what these simple representations miss.',
    why: `Every language technology — spam filters, search engines, chatbots, LLMs — starts by turning text into numbers. The classic representations (bag-of-words and TF-IDF) are still widely used for search, classification baselines and keyword extraction, and they power the "sparse" half of modern hybrid search in RAG systems. Seeing exactly what they capture (which words appear, how distinctive they are) and what they throw away (order, meaning, context) explains why embeddings and transformers were needed.`,
    objectives: [
      'Tokenize text and build a vocabulary',
      'Represent documents as bag-of-words count vectors and n-grams',
      'Compute TF-IDF weights and explain why rare words get higher weight',
      'Use cosine similarity on TF-IDF vectors for simple search, and list the limitations'
    ],
    sections: [
      { type: 'text', title: 'From text to tokens', md: `A model can't read characters; we first split text into **tokens** and give each distinct token an integer id. The simplest tokenizer lowercases and splits on whitespace (and maybe punctuation):

\`\`\`text
"The cat sat on the mat."  ->  ["the", "cat", "sat", "on", "the", "mat"]
\`\`\`

The set of distinct tokens is the **vocabulary**. Real systems must decide how to handle punctuation, numbers, casing, accents and unseen words — and modern LLMs use **subword** tokenizers (see *Subword Tokenization & BPE*), but the idea is the same: text → sequence of ids.` },
      { type: 'text', title: 'Bag-of-words and n-grams', md: `A **bag-of-words** (BoW) vector has one entry per vocabulary word: how many times that word appears in the document. Stacking documents gives a **document-term matrix** (documents × vocabulary), usually very **sparse** (mostly zeros).

| | the | cat | dog | sat | on | mat | log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| "the cat sat on the mat" | 2 | 1 | 0 | 1 | 1 | 1 | 0 |
| "the dog sat on the log" | 2 | 0 | 1 | 1 | 1 | 0 | 1 |

BoW forgets **word order**: "dog bites man" and "man bites dog" look identical. **n-grams** recover a little local order by also counting pairs (bigrams: "dog bites", "bites man") or triples. The vocabulary grows quickly, so n-grams are usually capped at 2 or 3.` },
      { type: 'text', title: 'TF-IDF: weighting distinctive words', md: `Raw counts over-weight words like "the" that appear everywhere and say nothing about a document's topic. **TF-IDF** multiplies two factors:

$$ tf(t, d) = count of t in d / number of tokens in d $$
$$ idf(t) = log( N / df(t) )          N = number of documents, df(t) = documents containing t $$
$$ tfidf(t, d) = tf(t, d) · idf(t) $$
:::where
t — a term (a word in the vocabulary)
d — a document
tf(t, d) — the term frequency: the share of d's tokens that are t
N — the number of documents in the corpus
df(t) — the document frequency: how many documents contain t at least once
idf(t) — the inverse document frequency: large for rare words, exactly 0 for a word in every document
log — the natural logarithm, which tames the ratio N / df(t) so very rare words don't dominate
tfidf(t, d) — the final weight of term t in document d
:::

- A word in **every** document has idf = log(1) = 0 → weight 0.
- A word in **few** documents has a high idf → it's distinctive for the documents that contain it.

Libraries use small variations (scikit-learn adds 1 inside and outside the log to avoid zeros, and normalises rows), but the intuition is the same. Comparing TF-IDF vectors with **cosine similarity** gives a decent search engine; the related **BM25** formula still powers keyword search in Elasticsearch and in hybrid RAG retrieval.` },
      { type: 'code', title: 'Bag-of-words, TF-IDF and a tiny search engine', code: `
import numpy as np

docs = ["the cat sat on the mat", "the dog sat on the log",
        "cats and dogs are pets", "the stock market fell today"]
tokens = [d.split() for d in docs]
vocab = sorted({w for t in tokens for w in t}); idx = {w: i for i, w in enumerate(vocab)}

counts = np.zeros((len(docs), len(vocab)))
for r, toks in enumerate(tokens):
    for w in toks:
        counts[r, idx[w]] += 1

tf = counts / counts.sum(axis=1, keepdims=True)
df = (counts > 0).sum(axis=0)
idf = np.log(len(docs) / df)
tfidf = tf * idf
print("vocabulary size:", len(vocab))
print("idf('the') =", round(float(idf[idx["the"]]), 3), "  idf('stock') =", round(float(idf[idx["stock"]]), 3))

def search(query):
    q = np.zeros(len(vocab))
    for w in query.split():
        if w in idx:
            q[idx[w]] += 1
    q = q / max(1, q.sum()) * idf
    sims = tfidf @ q / (np.linalg.norm(tfidf, axis=1) * np.linalg.norm(q) + 1e-12)
    return [(round(float(s), 3), d) for s, d in sorted(zip(sims, docs), reverse=True)[:2]]

print(search("a cat on a mat"))
print(search("market"))
` },
      { type: 'check', question: { type: 'mcq', q: 'A word appears in every document of the corpus. What is its IDF (with idf = log(N/df))?', options: ['1', '0', 'N', 'It is undefined'], answer: 1, explain: 'log(N/N) = log(1) = 0 — a word everywhere carries no distinguishing information.' } },
      { type: 'text', title: 'What these representations miss', md: `Bag-of-words and TF-IDF are fast, interpretable and strong baselines, but:
- **No word order** (beyond short n-grams): "not good" vs "good".
- **No meaning**: "car" and "automobile" are as unrelated as "car" and "banana"; synonyms don't match.
- **No context**: "bank" in "river bank" and "bank account" is the same column.
- **Huge sparse vectors** whose size grows with the vocabulary.

Dense **embeddings** fix similarity; **language models** and **transformers** add order and context. Still, in production search, combining TF-IDF/BM25 keyword matching with embedding search ("hybrid search") often beats either alone, because exact keywords (product codes, names) matter.` },
      { type: 'check', question: { type: 'tf', q: 'Bag-of-words vectors for "dog bites man" and "man bites dog" are identical.', answer: true, explain: 'Same words, same counts — order is lost.' } }
    ],
    when: `- **TF-IDF + logistic regression / linear SVM**: a strong, fast baseline for text classification with modest data.
- **BM25 / TF-IDF search**: keyword retrieval, often combined with embeddings (hybrid search) in RAG.
- **Use embeddings or transformers** when meaning, paraphrase, word order or context matter.`,
    pitfalls: `- Fitting the vocabulary and IDF on all data (leakage) — fit on training documents only.
- Forgetting consistent preprocessing (lowercasing, punctuation) at query time.
- Exploding vocabulary with n-grams and rare typos — set minimum document frequencies.
- Expecting synonyms to match.`,
    takeaways: [
      'Text → tokens → ids; the vocabulary is the set of distinct tokens.',
      'Bag-of-words counts words per document and ignores order; n-grams add local order.',
      'TF-IDF = term frequency × log(N / document frequency): common words get low weight.',
      'Cosine similarity of TF-IDF vectors is a simple, effective keyword search.',
      'They miss meaning and context — the motivation for embeddings and transformers.'
    ],
    quiz: [
      { type: 'fill', q: 'In 100 documents, the word "quantum" appears in 10. What is its IDF using natural log, log(N/df)? (2 decimals)', answer: ['2.30', '2.3', '2.303', 'ln10', 'ln 10'], display: 'ln 10 ≈ 2.30', explain: 'log(100/10) = ln 10 ≈ 2.30.' },
      { type: 'mcq', q: 'Which information does a bag-of-words representation keep?', options: ['Word order', 'Which words appear and how often', 'Sentence structure', 'Word meanings'], answer: 1, explain: 'Only counts per vocabulary word.' },
      { type: 'mcq', q: 'What are bigrams?', options: ['Two-letter words', 'Consecutive pairs of tokens', 'Words that appear twice', 'Two documents'], answer: 1, explain: 'E.g. "new york", "not good".' },
      { type: 'tf', q: 'TF-IDF search will match a query "automobile" with a document that only uses the word "car".', answer: false, explain: 'Different tokens are different dimensions; synonyms don\'t match without embeddings or query expansion.' },
      { type: 'mcq', q: 'Why is "hybrid search" (keywords + embeddings) popular in RAG?', options: ['Keywords are always better', 'Embeddings capture meaning while keyword matching nails exact terms like names and codes', 'It is cheaper than either', 'It removes the need for chunking'], answer: 1, explain: 'The two retrieval styles fail in different ways, so combining them is robust.' }
    ],
    problems: [
      {
        id: 'bag-of-words', title: 'Bag-of-Words Matrix', difficulty: 'easy',
        prompt: `Implement:

1. \`tokenize(doc)\` → \`doc.lower().split()\`.
2. \`build_vocab(docs)\` → sorted list of all distinct tokens.
3. \`bow_matrix(docs, vocab)\` → NumPy array (n_docs, len(vocab)) of token counts; tokens not in \`vocab\` are ignored.

\`\`\`text
docs = ["The cat sat", "the cat the hat"]
build_vocab(docs)        ->  ["cat", "hat", "sat", "the"]
bow_matrix(docs, vocab)  ->  [[1, 0, 1, 1], [1, 1, 0, 2]]
\`\`\``,
        starter: `
import numpy as np

def tokenize(doc):
    # TODO
    pass

def build_vocab(docs):
    # TODO
    pass

def bow_matrix(docs, vocab):
    # TODO
    pass
`,
        solution: `
import numpy as np

def tokenize(doc):
    return doc.lower().split()

def build_vocab(docs):
    return sorted({w for d in docs for w in tokenize(d)})

def bow_matrix(docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for w in tokenize(d):
            j = index.get(w)
            if j is not None:
                M[r, j] += 1
    return M
`,
        hints: ['A set comprehension collects distinct tokens; wrap it in `sorted()`.', 'Map each word to its column with a dict, then count.'],
        explain: `This is CountVectorizer in miniature. In practice the matrix is stored sparsely (most entries are 0), which matters once the vocabulary has tens of thousands of words.`,
        tests: [
          { call: 'build_vocab(["The cat sat", "the cat the hat"])', expect: '["cat", "hat", "sat", "the"]' },
          { call: 'bow_matrix(["The cat sat", "the cat the hat"], ["cat", "hat", "sat", "the"])', expect: 'np.array([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 2.0]])' },
          { name: 'unknown words ignored', call: 'bow_matrix(["a b c d"], ["b", "d"])', expect: 'np.array([[1.0, 1.0]])', hidden: true },
          { name: 'random corpus', setup: 'rng = np.random.default_rng(0)\nwords = ["red", "green", "blue", "cat", "dog"]\ndocs = [" ".join(rng.choice(words, size=6)) for _ in range(5)]', call: 'bow_matrix(docs, build_vocab(docs))', hidden: true }
        ]
      },
      {
        id: 'tf-idf', title: 'TF-IDF from Scratch', difficulty: 'medium',
        prompt: `Implement \`tfidf(docs)\` → \`(vocab, matrix)\` using the provided \`build_vocab\` and \`bow_matrix\`:

- tf = counts / (number of tokens in that document) — row-normalised counts;
- df = number of documents containing each word;
- idf = ln(N / df) (natural log, no smoothing);
- matrix = tf × idf (broadcast over rows).

\`\`\`text
tfidf(["a b", "a c"])  ->  vocab ["a", "b", "c"],
                          matrix [[0, 0.5·ln2, 0], [0, 0, 0.5·ln2]]
\`\`\``,
        starter: `
import numpy as np

def tokenize(doc):
    return doc.lower().split()

def build_vocab(docs):
    return sorted({w for d in docs for w in tokenize(d)})

def bow_matrix(docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for w in tokenize(d):
            if w in index:
                M[r, index[w]] += 1
    return M

def tfidf(docs):
    # TODO
    pass
`,
        solution: `
import numpy as np

def tokenize(doc):
    return doc.lower().split()

def build_vocab(docs):
    return sorted({w for d in docs for w in tokenize(d)})

def bow_matrix(docs, vocab):
    index = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for w in tokenize(d):
            if w in index:
                M[r, index[w]] += 1
    return M

def tfidf(docs):
    vocab = build_vocab(docs)
    counts = bow_matrix(docs, vocab)
    tf = counts / counts.sum(axis=1, keepdims=True)
    df = (counts > 0).sum(axis=0)
    idf = np.log(len(docs) / df)
    return vocab, tf * idf
`,
        hints: ['Row-normalise with `counts.sum(axis=1, keepdims=True)`.', '`(counts > 0).sum(axis=0)` counts documents per word.', 'Multiply the (n_docs, V) tf matrix by the (V,) idf vector.'],
        explain: `The broadcasting multiply applies each word's idf to its whole column. Note how "a" (present in every document) gets weight exactly 0 — it can't help distinguish documents.`,
        tests: [
          { call: 'tfidf(["a b", "a c"])', expect: '(["a", "b", "c"], np.array([[0.0, 0.5 * np.log(2), 0.0], [0.0, 0.0, 0.5 * np.log(2)]]))' },
          { name: 'repeated words', call: 'tfidf(["x x y", "y z"])[1]', expect: 'np.array([[2 / 3 * np.log(2), 0.0, 0.0], [0.0, 0.0, 0.5 * np.log(2)]])' },
          { name: 'larger corpus', setup: 'docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are pets", "the stock market fell today"]', call: 'tfidf(docs)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Text feature extraction', url: 'https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction' },
      { title: 'Jurafsky & Martin — Speech and Language Processing (free draft)', url: 'https://web.stanford.edu/~jurafsky/slp3/' }
    ]
  },

  // =====================================================================================
  {
    id: 'language-models', branch: 'llm', title: 'Language Models & N-grams', short: 'Language Models', level: 2, minutes: 40, prereqs: ['text-representation', 'probability', 'information-theory', 'mle-map'],
    tags: ['language model', 'n-gram', 'bigram', 'next token prediction', 'perplexity', 'smoothing', 'sampling'],
    summary: 'A language model assigns probabilities to text by predicting the next token. Counting-based n-gram models, smoothing, generating text, and measuring quality with perplexity.',
    why: `"Predict the next token" is the entire training objective of GPT, Claude and Llama. Before neural networks, the same idea was implemented by **counting**: how often does each word follow the previous one? Building an n-gram model by hand makes the concepts concrete — conditional probabilities, the chain rule for sequences, sampling to generate text, perplexity to evaluate — and shows exactly what neural language models improve: they generalize to contexts never seen verbatim and use far longer context.`,
    objectives: [
      'Define a language model and factor the probability of a sentence with the chain rule',
      'Estimate bigram probabilities by counting and apply add-k smoothing',
      'Generate text by sampling from the model',
      'Compute perplexity and explain the limits of n-gram models'
    ],
    sections: [
      { type: 'text', title: 'What a language model is', md: `A **language model** assigns a probability to any sequence of tokens. By the chain rule of probability, that can always be written as a product of next-token predictions:

$$ P(w₁, w₂, …, wₙ) = P(w₁) · P(w₂ | w₁) · P(w₃ | w₁, w₂) · … · P(wₙ | w₁…wₙ₋₁) $$
:::where
w₁, …, wₙ — the tokens of a sequence, in order
P(w₁, …, wₙ) — the probability of the whole sequence
P(wₜ | w₁…wₜ₋₁) — the probability of token t given all the tokens before it: one next-token prediction
:::

So a language model only needs to answer one question, repeatedly: **given the text so far, what is the probability distribution over the next token?**

With that you can:
- **score** text (is this sentence likely? useful for speech recognition, spelling correction);
- **generate** text by sampling a token, appending it, and repeating;
- **complete** prompts — which is all a chatbot does, with a lot of training to make the completions helpful.` },
      { type: 'text', title: 'n-gram models: counting', md: `Conditioning on the entire history is impossible to estimate by counting — almost every long history is unique. The **n-gram** (Markov) assumption keeps only the last n − 1 tokens:

$$ bigram (n = 2):   P(wₜ | w₁…wₜ₋₁) ≈ P(wₜ | wₜ₋₁) $$
:::where
wₜ — the token at position t
w₁…wₜ₋₁ — the entire history before it
wₜ₋₁ — just the previous token, which is all a bigram model keeps
≈ — "approximately": the Markov assumption throws away the older history
n — the size of the n-gram; an n-gram model conditions on the last n − 1 tokens
:::

Estimate it by counting in a corpus (maximum likelihood):

$$ P(cat | the) = count("the cat") / count("the" followed by anything) $$
:::where
count("the cat") — how many times the pair "the cat" occurs in the training corpus
count("the" followed by anything) — how many times "the" occurs with any next token
P(cat | the) — the estimated probability that "cat" follows "the" (the maximum-likelihood estimate)
:::

**The zero problem.** Any pair never seen in training gets probability 0, which makes the whole sentence's probability 0. **Add-k (Laplace) smoothing** adds a small count k to every possible pair:

$$ P(b | a) = (count(a b) + k) / (count(a •) + k·V) $$
:::where
a, b — a token and the token that follows it
count(a b) — how often the pair "a b" occurs in training
count(a •) — how often a is followed by anything (the • stands for any token)
k — the small pseudo-count added to every possible pair (k = 1 is Laplace smoothing)
V — the vocabulary size; adding k·V to the denominator keeps each row of probabilities summing to 1
:::

(More refined methods — Kneser-Ney, back-off — were the state of the art until neural LMs.)` },
      { type: 'code', title: 'A character-level bigram model that writes', code: `
import numpy as np
text = "the cat sat on the mat. the cat ate the rat. a rat sat on a hat. the hat is flat."
chars = sorted(set(text)); idx = {c: i for i, c in enumerate(chars)}

counts = np.ones((len(chars), len(chars)))          # add-one smoothing
for a, b in zip(text, text[1:]):
    counts[idx[a], idx[b]] += 1
P = counts / counts.sum(axis=1, keepdims=True)      # P[next | current], each row sums to 1

rng = np.random.default_rng(0)
c, out = "t", ["t"]
for _ in range(70):                                 # generate by sampling the next character
    c = chars[rng.choice(len(chars), p=P[idx[c]])]
    out.append(c)
print("sampled text:", "".join(out))

def perplexity(s):
    logp = [np.log(P[idx[a], idx[b]]) for a, b in zip(s, s[1:])]
    return float(np.exp(-np.mean(logp)))
print("perplexity of 'the cat sat on the mat':", round(perplexity("the cat sat on the mat"), 2))
print("perplexity of 'tca hta mso nhe':       ", round(perplexity("tca hta mso nhe"), 2))
`, after: `The model knows only which character tends to follow which, so the output has plausible letter pairs but no words or meaning. Natural text gets a much lower perplexity than scrambled text.` },
      { type: 'check', question: { type: 'mcq', q: 'In a bigram model, "the" is followed by "cat" 30 times and by other words 70 times. What is P(cat | the) without smoothing?', options: ['0.3', '0.7', '30', '0.43'], answer: 0, explain: '30 / (30 + 70) = 0.3.' } },
      { type: 'text', title: 'Perplexity', md: `How good is a language model? Measure how surprised it is by real held-out text. With p₁, …, p_N the probabilities the model assigned to each actual next token:

$$ perplexity = exp( − (1/N) Σ log pᵢ ) $$
:::where
N — the number of tokens in the held-out text
pᵢ — the probability the model gave to the token that actually came next at position i
−log pᵢ — the model's surprise at that token (natural log)
(1/N) Σ — the average over all N tokens, which is the cross-entropy in nats
exp — e to that power, which turns the average surprise back into a "number of choices"
:::

It is the exponentiated average cross-entropy (see *Information Theory*): roughly **how many equally likely choices the model is torn between** at each step. Lower is better. A uniform guess over a 50,000-word vocabulary has perplexity 50,000; classic n-gram models on news text reached about 100–200; modern LLMs reach single digits on similar text (with subword tokens).` },
      { type: 'text', title: 'Why neural language models won', md: `n-gram models hit hard limits:
- **Sparsity** — most long n-grams are never seen, so counts can't generalize; "the purple cat" teaches nothing about "the violet cat".
- **Short context** — a 5-gram model sees 4 tokens; real meaning depends on paragraphs.
- **Size** — the table of counts explodes with n.

**Neural language models** replace the count table with a network: embeddings let similar words share statistical strength, and architectures with long context (RNNs, then transformers) condition on thousands of tokens. A GPT is still a next-token predictor trained with exactly the cross-entropy/perplexity you just computed — but with a learned function instead of a lookup table, and trained on trillions of tokens.` },
      { type: 'check', question: { type: 'tf', q: 'A lower perplexity on held-out text means a better language model.', answer: true, explain: 'It means the model assigned higher probability to what actually came next.' } }
    ],
    when: `- n-gram models: tiny, fast, interpretable baselines; autocomplete on constrained domains; teaching. (They were used in production speech recognition and machine translation for decades.)
- Neural/transformer LMs for anything needing real fluency, long context or generalization.
- Perplexity: comparing language models **with the same tokenizer** on the same held-out text.`,
    pitfalls: `- Zero probabilities without smoothing.
- Comparing perplexities across different tokenizers or datasets.
- Evaluating on text that overlaps with training text.
- Confusing "likely" with "true": a language model scores plausibility, not correctness.`,
    takeaways: [
      'A language model gives P(next token | previous tokens); the chain rule turns that into the probability of any text.',
      'n-gram models estimate it by counting, using only the last n − 1 tokens.',
      'Smoothing prevents unseen n-grams from getting probability zero.',
      'Generation = repeatedly sample the next token and append it.',
      'Perplexity = exp(mean negative log-probability) — lower is better.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the bigram assumption say?', options: ['Every word is independent', 'The next word depends only on the previous word', 'Sentences have two words', 'Only pairs of documents matter'], answer: 1, explain: 'P(wₜ | all history) ≈ P(wₜ | wₜ₋₁).' },
      { type: 'fill', q: 'A model assigns probability 0.25 to every actual next token in a test text. What is its perplexity?', answer: ['4', '4.0'], explain: 'exp(−log 0.25) = 1/0.25 = 4.' },
      { type: 'mcq', q: 'Why is smoothing needed in n-gram models?', options: ['To make the model faster', 'Unseen n-grams would otherwise get probability 0 and zero out whole sentences', 'To shorten the context', 'To remove rare words'], answer: 1, explain: 'One zero factor makes the product zero.' },
      { type: 'mcq', q: 'What is the training objective of GPT-style models?', options: ['Classify sentiment', 'Predict the next token (cross-entropy)', 'Translate between languages', 'Count n-grams'], answer: 1, explain: 'Next-token prediction on huge text corpora.' },
      { type: 'tf', q: 'A language model that assigns high probability to a sentence guarantees the sentence is factually true.', answer: false, explain: 'It measures how plausible text is under its training distribution, not truth.' }
    ],
    problems: [
      {
        id: 'bigram-model', title: 'A Bigram Language Model', difficulty: 'medium',
        prompt: `Implement \`bigram_probs(tokens)\` returning a nested dict: \`probs[prev][nxt]\` = count of the pair (prev, nxt) divided by the number of pairs that start with prev (maximum likelihood, no smoothing). Only include pairs that occur.

Then implement \`sentence_prob(tokens, probs)\` → the product of P(tᵢ | tᵢ₋₁) over consecutive pairs of a new sentence (return 0.0 if any pair is unseen).

\`\`\`text
tokens = "the cat sat the cat ran".split()
bigram_probs(tokens) -> {"the": {"cat": 1.0}, "cat": {"sat": 0.5, "ran": 0.5}, "sat": {"the": 1.0}}
\`\`\``,
        starter: `
def bigram_probs(tokens):
    # TODO: count pairs, then normalise each row
    pass

def sentence_prob(tokens, probs):
    # TODO
    pass
`,
        solution: `
def bigram_probs(tokens):
    counts = {}
    for a, b in zip(tokens, tokens[1:]):
        counts.setdefault(a, {})
        counts[a][b] = counts[a].get(b, 0) + 1
    return {a: {b: c / sum(nxt.values()) for b, c in nxt.items()} for a, nxt in counts.items()}

def sentence_prob(tokens, probs):
    p = 1.0
    for a, b in zip(tokens, tokens[1:]):
        p *= probs.get(a, {}).get(b, 0.0)
    return p
`,
        hints: ['`zip(tokens, tokens[1:])` yields consecutive pairs.', 'Use `dict.setdefault` and `.get(..., 0) + 1` to count.', 'Normalise each prev-row by the sum of its counts.'],
        explain: `The nested dict is a sparse version of the (V × V) transition matrix — only observed pairs are stored. The zero for unseen pairs in \`sentence_prob\` is the sparsity problem that smoothing (and ultimately neural models) solve.`,
        tests: [
          { call: 'bigram_probs("the cat sat the cat ran".split())', expect: '{"the": {"cat": 1.0}, "cat": {"sat": 0.5, "ran": 0.5}, "sat": {"the": 1.0}}' },
          { call: 'sentence_prob("the cat ran".split(), bigram_probs("the cat sat the cat ran".split()))', expect: '0.5' },
          { name: 'unseen pair gives 0', call: 'sentence_prob("cat the".split(), bigram_probs("the cat sat".split()))', expect: '0.0', hidden: true },
          { name: 'random token stream', setup: 'rng = np.random.default_rng(0)\ntoks = [str(t) for t in rng.choice(["a", "b", "c", "d"], size=60)]', call: 'bigram_probs(toks)', hidden: true }
        ]
      },
      {
        id: 'perplexity', title: 'Perplexity', difficulty: 'easy',
        prompt: `Implement \`perplexity(probs)\`: given the list of probabilities a model assigned to each actual next token, return exp(−mean(ln p)).

\`\`\`text
perplexity([0.5, 0.5])        ->  2.0
perplexity([0.25] * 4)        ->  4.0
\`\`\`

Also implement \`perplexity_from_loss(loss)\` → exp(loss), for a mean cross-entropy loss in nats.`,
        starter: `
import math

def perplexity(probs):
    # TODO
    pass

def perplexity_from_loss(loss):
    # TODO
    pass
`,
        solution: `
import math

def perplexity(probs):
    return math.exp(-sum(math.log(p) for p in probs) / len(probs))

def perplexity_from_loss(loss):
    return math.exp(loss)
`,
        hints: ['Average the natural logs, negate, exponentiate.'],
        explain: `Averaging log-probabilities (instead of multiplying probabilities) keeps the computation stable and makes perplexity independent of text length — a per-token measure. During LLM training you will see loss curves in nats; exp(loss) converts them to perplexity.`,
        tests: [
          { call: 'perplexity([0.5, 0.5])', expect: '2.0' },
          { call: 'perplexity([0.25] * 4)', expect: '4.0' },
          { call: 'perplexity_from_loss(2.0)', expect: 'math.exp(2.0)' },
          { name: 'mixed probabilities', call: 'perplexity([0.9, 0.1])', expect: 'math.exp(-(math.log(0.9) + math.log(0.1)) / 2)', hidden: true },
          { name: 'random probabilities', setup: 'rng = np.random.default_rng(1)\nps = rng.uniform(0.01, 1.0, 50).tolist()', call: 'perplexity(ps)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Jurafsky & Martin — N-gram Language Models (chapter 3)', url: 'https://web.stanford.edu/~jurafsky/slp3/3.pdf' },
      { title: 'Karpathy — makemore (building language models from scratch)', url: 'https://github.com/karpathy/makemore' }
    ]
  },

  // =====================================================================================
  {
    id: 'tokenization', branch: 'llm', title: 'Subword Tokenization & Byte-Pair Encoding', short: 'Tokenization & BPE', level: 2, minutes: 40, prereqs: ['text-representation'],
    tags: ['tokenizer', 'bpe', 'subword', 'byte-level', 'sentencepiece', 'tokens', 'vocabulary'],
    summary: 'How LLMs split text into tokens: why subwords beat words and characters, the BPE training algorithm, byte-level BPE, and the practical consequences of tokenization.',
    why: `An LLM never sees letters or words — only token ids produced by its **tokenizer**. Tokenization decides the vocabulary size, how long sequences are, what you pay per API call, how much text fits in the context window, and explains many odd LLM behaviours: trouble counting letters in "strawberry", inconsistent arithmetic, and higher cost for non-English languages. Byte-Pair Encoding (BPE) — or a close relative — is used by GPT, Llama and most modern models.`,
    objectives: [
      'Explain the trade-offs of word-level, character-level and subword tokenization',
      'Run the BPE training algorithm: count pairs, merge the most frequent, repeat',
      'Encode new text with learned merges and explain byte-level BPE',
      'Describe practical consequences of tokenization for LLM cost, context and behaviour'
    ],
    sections: [
      { type: 'text', title: 'Words, characters or something in between?', md: `| Strategy | Vocabulary | Sequence length | Problems |
| --- | --- | --- | --- |
| **Words** | huge (millions with names, typos, inflections) | short | unknown words at test time; "run/runs/running" unrelated |
| **Characters** | tiny (~100s) | very long | the model must learn spelling from scratch; long sequences are expensive for attention |
| **Subwords** | medium (~30k–200k) | moderate | the sweet spot |

Subword tokenizers keep frequent words whole ("the", "model") and split rare ones into meaningful pieces ("tokenization" → "token" + "ization"; "unhappiness" → "un" + "happiness"). Nothing is ever truly unknown: in the worst case, a word splits into single characters (or bytes).` },
      { type: 'text', title: 'Byte-Pair Encoding (BPE)', md: `BPE **learns** its vocabulary from a training corpus, bottom-up:

1. Start with each word as a sequence of characters, plus an end-of-word marker (here \`</w>\`), and count word frequencies.
2. Count every **adjacent pair** of symbols across the corpus (weighted by word frequency).
3. **Merge** the most frequent pair into a new symbol everywhere (e.g. "e" + "s" → "es").
4. Record the merge and repeat until the vocabulary reaches the target size (number of merges).

\`\`\`text
corpus: low ×5, lower ×2, newest ×6, widest ×3
start:  l o w </w>   l o w e r </w>   n e w e s t </w>   w i d e s t </w>
merge 1: e + s -> es         (9 times: newest ×6 + widest ×3)
merge 2: es + t -> est       (9 times)
merge 3: est + </w> -> est</w>
merge 4: l + o -> lo  ...
\`\`\`

To **encode** new text, apply the learned merges in the same order. Frequent words end up as single tokens; rare words as several.` },
      { type: 'demo', title: 'Interactive: BPE merges step by step', md: `Type any text and keep merging the most frequent pair. Watch the vocabulary grow and the token count shrink.`, demo: 'bpe' },
      { type: 'code', title: 'Training BPE from scratch', code: `
from collections import Counter

text = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
words = Counter(tuple(w) + ("</w>",) for w in text.split())   # word (as symbols) -> frequency

def pair_counts(words):
    counts = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts

def merge(words, pair):
    out = {}
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new.append(symbols[i] + symbols[i + 1]); i += 2
            else:
                new.append(symbols[i]); i += 1
        out[tuple(new)] = freq
    return out

for step in range(8):
    counts = pair_counts(words)
    best = min(counts, key=lambda p: (-counts[p], p))      # most frequent; ties -> alphabetical
    words = merge(words, best)
    print(f"merge {step + 1}: {best[0]} + {best[1]} -> {best[0] + best[1]}  ({counts[best]}x)")
print("final segmentation:", [" ".join(s) for s in words])
` },
      { type: 'check', question: { type: 'mcq', q: 'In BPE training, which pair is merged at each step?', options: ['A random pair', 'The most frequent adjacent pair of symbols in the corpus', 'The longest word', 'The rarest pair'], answer: 1, explain: 'Greedy frequency-based merging builds tokens for common character sequences.' } },
      { type: 'text', title: 'Byte-level BPE and other tokenizers', md: `**Byte-level BPE** (GPT-2 and later) starts from the 256 possible **bytes** of UTF-8 instead of characters. Any string — emoji, Chinese, code, binary noise — can be encoded without an "unknown" token. GPT-4-era tokenizers have ~100k tokens; Llama 3 uses ~128k.

Relatives: **WordPiece** (BERT) picks merges by likelihood rather than raw frequency; the **Unigram** model (in **SentencePiece**, used by T5 and many multilingual models) starts with a large vocabulary and prunes it. Tokenizers also define **special tokens**: beginning/end of sequence, padding, and chat-format markers that separate system, user and assistant messages.` },
      { type: 'text', title: 'Practical consequences', md: `- **Cost and limits are in tokens.** English averages roughly 4 characters (≈ ¾ of a word) per token; code and many other languages need more tokens for the same content — so they cost more and fit less in the context window.
- **Strange blind spots.** "strawberry" may be one or a few tokens, so the model never directly "sees" its letters — counting r's is surprisingly hard. Numbers split into irregular chunks ("1234" → "123" + "4"), which hurts arithmetic.
- **Leading spaces matter.** " hello" and "hello" are usually different tokens.
- **The tokenizer is part of the model.** You must use the exact tokenizer a model was trained with; perplexities of models with different tokenizers aren't comparable.
- **Glitch tokens.** Rare tokens that barely appeared in training can trigger bizarre outputs.` },
      { type: 'check', question: { type: 'tf', q: 'A byte-level BPE tokenizer can encode any input text without needing an "unknown" token.', answer: true, explain: 'Every string is a sequence of bytes, and all 256 bytes are in the base vocabulary.' } }
    ],
    when: `- Always use the tokenizer that ships with the model you use (Hugging Face \`AutoTokenizer\`, tiktoken for OpenAI models, etc.).
- Count tokens (not characters) when budgeting context windows and API costs.
- Train your own tokenizer only when pretraining a model from scratch, especially for a new language or domain (code, chemistry, DNA).`,
    pitfalls: `- Estimating cost or context use from character or word counts.
- Mixing tokenizers between training and inference.
- Forgetting special tokens and chat templates when fine-tuning.
- Assuming the model can see individual characters or digits.`,
    takeaways: [
      'Subword tokenization balances vocabulary size and sequence length; nothing is ever unknown.',
      'BPE repeatedly merges the most frequent adjacent pair; encoding replays the merges in order.',
      'Byte-level BPE (GPT-2+) works on UTF-8 bytes and handles any input.',
      'Token counts drive cost, context limits and some odd model behaviours.',
      'A model must be used with its own tokenizer.'
    ],
    quiz: [
      { type: 'mcq', q: 'Why don\'t modern LLMs use whole words as tokens?', options: ['Words are too short', 'The vocabulary would be enormous and unseen words couldn\'t be represented', 'Words can\'t be embedded', 'GPUs cannot process words'], answer: 1, explain: 'Subwords keep the vocabulary bounded and handle any new word.' },
      { type: 'mcq', q: 'Roughly how many characters of English text does one token represent on average?', options: ['1', 'about 4', 'about 20', 'exactly one word'], answer: 1, explain: 'A common rule of thumb is ~4 characters (~0.75 words) per token.' },
      { type: 'tf', q: 'The same text always takes the same number of tokens regardless of the model.', answer: false, explain: 'Different tokenizers split text differently.' },
      { type: 'mcq', q: 'Why might an LLM miscount the letters in a word?', options: ['It cannot do arithmetic at all', 'It sees subword tokens, not individual characters', 'Its context window is too small', 'It was trained only on numbers'], answer: 1, explain: 'Characters inside a token are not directly visible to the model.' },
      { type: 'order', q: 'Order one BPE training iteration.', items: ['Count all adjacent symbol pairs (weighted by word frequency)', 'Pick the most frequent pair', 'Merge that pair everywhere into a new symbol', 'Record the merge rule'], explain: 'Count, choose, merge, record — then repeat.' }
    ],
    problems: [
      {
        id: 'bpe-pair-counts', title: 'BPE: Count Symbol Pairs', difficulty: 'easy',
        prompt: `In BPE, the corpus is a dict mapping each word (a **tuple of symbols**) to its frequency, e.g.
\`{("l", "o", "w", "</w>"): 5, ("l", "o", "w", "e", "r", "</w>"): 2}\`.

Implement \`get_pair_counts(words)\` → a dict mapping each adjacent pair \`(a, b)\` to its total count, weighted by word frequency.

\`\`\`text
get_pair_counts({("l", "o", "w", "</w>"): 5, ("l", "o", "w", "e", "r", "</w>"): 2})
-> {("l","o"): 7, ("o","w"): 7, ("w","</w>"): 5, ("w","e"): 2, ("e","r"): 2, ("r","</w>"): 2}
\`\`\``,
        starter: `
def get_pair_counts(words):
    counts = {}
    # TODO
    return counts
`,
        solution: `
def get_pair_counts(words):
    counts = {}
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + freq
    return counts
`,
        hints: ['`zip(symbols, symbols[1:])` gives adjacent pairs.', 'Add the word\'s frequency, not 1.'],
        explain: `Weighting by frequency means a pair inside a very common word counts many times — so BPE prioritises merges that shorten the most text overall.`,
        tests: [
          { call: 'get_pair_counts({("l", "o", "w", "</w>"): 5, ("l", "o", "w", "e", "r", "</w>"): 2})', expect: '{("l", "o"): 7, ("o", "w"): 7, ("w", "</w>"): 5, ("w", "e"): 2, ("e", "r"): 2, ("r", "</w>"): 2}' },
          { name: 'repeated pair inside one word', call: 'get_pair_counts({("a", "a", "a"): 3})', expect: '{("a", "a"): 6}', hidden: true },
          { name: 'single-symbol word has no pairs', call: 'get_pair_counts({("x",): 10})', expect: '{}', hidden: true }
        ]
      },
      {
        id: 'bpe-merge', title: 'BPE: Merge a Pair', difficulty: 'medium',
        prompt: `Implement \`merge_pair(words, pair)\`: return a new words dict where every occurrence of the adjacent symbols \`pair = (a, b)\` is replaced by the single symbol \`a + b\`. Scan each word left to right; occurrences don't overlap (after merging at position i, continue at i + 2).

\`\`\`text
merge_pair({("n", "e", "w", "e", "s", "t", "</w>"): 6}, ("e", "s"))
-> {("n", "e", "w", "es", "t", "</w>"): 6}

merge_pair({("a", "a", "a"): 1}, ("a", "a"))  ->  {("aa", "a"): 1}
\`\`\``,
        starter: `
def merge_pair(words, pair):
    out = {}
    # TODO
    return out
`,
        solution: `
def merge_pair(words, pair):
    out = {}
    a, b = pair
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        key = tuple(new)
        out[key] = out.get(key, 0) + freq
    return out
`,
        hints: ['Use a while loop with an index so you can skip ahead by 2 after a merge.', 'Build a new tuple for each word; keep its frequency.'],
        explain: `Left-to-right, non-overlapping replacement makes "aaa" become ("aa", "a"), matching how real BPE implementations apply a merge. Summing frequencies on collision keeps counts correct if two words become identical.`,
        tests: [
          { call: 'merge_pair({("n", "e", "w", "e", "s", "t", "</w>"): 6}, ("e", "s"))', expect: '{("n", "e", "w", "es", "t", "</w>"): 6}' },
          { call: 'merge_pair({("a", "a", "a"): 1}, ("a", "a"))', expect: '{("aa", "a"): 1}' },
          { name: 'several words, some unaffected', call: 'merge_pair({("l", "o", "w", "</w>"): 5, ("w", "i", "d", "e", "</w>"): 2}, ("l", "o"))', expect: '{("lo", "w", "</w>"): 5, ("w", "i", "d", "e", "</w>"): 2}', hidden: true },
          { name: 'pair appears twice in one word', call: 'merge_pair({("a", "b", "c", "a", "b"): 4}, ("a", "b"))', expect: '{("ab", "c", "ab"): 4}', hidden: true }
        ]
      },
      {
        id: 'bpe-train', title: 'Train a BPE Tokenizer', difficulty: 'hard',
        prompt: `Put it together. Implement \`train_bpe(text, num_merges)\` → list of merges (each a tuple \`(a, b)\`):

1. Words = whitespace-split \`text\`; represent each as \`tuple(word) + ("</w>",)\`; count frequencies.
2. Repeat \`num_merges\` times (stop early if no pairs remain):
   - count pairs (\`get_pair_counts\`);
   - choose the most frequent pair; **ties → the lexicographically smallest pair** (e.g. \`min(counts, key=lambda p: (-counts[p], p))\`);
   - merge it (\`merge_pair\`) and append it to the list.

Then implement \`encode(word, merges)\`: start from \`list(word) + ["</w>"]\` and apply each merge in order (non-overlapping, left to right); return the list of symbols.

Helpers \`get_pair_counts\` and \`merge_pair\` are provided.`,
        starter: `
from collections import Counter

def get_pair_counts(words):
    counts = {}
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + freq
    return counts

def merge_pair(words, pair):
    out = {}
    a, b = pair
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(symbols[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out

def train_bpe(text, num_merges):
    # TODO
    pass

def encode(word, merges):
    # TODO
    pass
`,
        solution: `
from collections import Counter

def get_pair_counts(words):
    counts = {}
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + freq
    return counts

def merge_pair(words, pair):
    out = {}
    a, b = pair
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(symbols[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out

def train_bpe(text, num_merges):
    words = dict(Counter(tuple(w) + ("</w>",) for w in text.split()))
    merges = []
    for _ in range(num_merges):
        counts = get_pair_counts(words)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        words = merge_pair(words, best)
        merges.append(best)
    return merges

def encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(symbols[i]); i += 1
        symbols = new
    return symbols
`,
        hints: ['`Counter(tuple(w) + ("</w>",) for w in text.split())` builds the initial corpus.', 'The key `(-count, pair)` sorts by highest count, then alphabetical.', 'Encoding replays the merges in the order they were learned.'],
        explain: `Training is a loop of count → pick → merge; the list of merges *is* the tokenizer. Encoding replays them in order, so frequent words collapse into one token while unseen words still decompose into known pieces — never an "unknown".`,
        tests: [
          { name: 'first merges on the classic corpus', call: 'train_bpe("low low low low low lower lower newest newest newest newest newest newest widest widest widest", 4)', expect: '[("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")]' },
          { name: 'encode a seen word', setup: 'merges = train_bpe("low low low low low lower lower newest newest newest newest newest newest widest widest widest", 10)', call: 'encode("newest", merges)' },
          { name: 'encode an unseen word with known pieces', setup: 'merges = train_bpe("low low low low low lower lower newest newest newest newest newest newest widest widest widest", 10)', call: 'encode("lowest", merges)', hidden: true },
          { name: 'stops when nothing is left to merge', call: 'train_bpe("ab ab", 10)', expect: '[("a", "b"), ("ab", "</w>")]', hidden: true },
          { name: 'random text', setup: 'rng = np.random.default_rng(0)\ntext = " ".join("".join(rng.choice(list("abcd"), size=rng.integers(2, 6))) for _ in range(40))', call: 'train_bpe(text, 12)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Sennrich et al. — Neural Machine Translation of Rare Words with Subword Units (BPE)', url: 'https://arxiv.org/abs/1508.07909' },
      { title: 'Karpathy — minbpe (minimal BPE implementation) ', url: 'https://github.com/karpathy/minbpe' },
      { title: 'OpenAI tiktoken', url: 'https://github.com/openai/tiktoken' }
    ]
  },

  // =====================================================================================
  {
    id: 'attention', branch: 'llm', title: 'Attention & Self-Attention', short: 'Attention', level: 2, minutes: 50, prereqs: ['embeddings', 'activation-functions', 'seq2seq'],
    tags: ['attention', 'self-attention', 'query', 'key', 'value', 'causal mask', 'softmax'],
    summary: 'The core idea behind transformers: every token builds a new representation as a weighted mix of all the tokens it attends to — queries, keys, values, scaled dot products, softmax, and the causal mask.',
    why: `Attention is the single most important idea in modern AI. It lets every token in a sequence look at every other token and decide, from the content itself, which ones matter: "it" can attend to "cat" twenty words back; a verb can find its subject; a code token can find the variable's definition. It replaced recurrence (no more one-step-at-a-time bottleneck), parallelizes beautifully on GPUs, and is the heart of every transformer — GPT, BERT, Llama, vision transformers and diffusion models.`,
    objectives: [
      'Explain attention as a soft, differentiable dictionary lookup with queries, keys and values',
      'Compute scaled dot-product attention step by step and explain the √d scaling',
      'Build self-attention with learned W_Q, W_K, W_V projections',
      'Apply a causal mask for language modelling and state attention\'s O(n²) cost'
    ],
    sections: [
      { type: 'text', title: 'The problem attention solves', md: `Meaning depends on context: in "The animal didn't cross the street because **it** was too tired", *it* refers to the animal; change "tired" to "wide" and *it* refers to the street. A good representation of "it" must pull in information from the right earlier word.

RNNs squeeze all earlier context into one hidden vector, passed step by step — distant information fades and nothing can be parallelized. **Attention** gives every token direct access to every other token, with weights computed from the content of the tokens themselves.` },
      { type: 'text', title: 'Queries, keys and values', md: `Think of a soft dictionary lookup. Each token produces three vectors:

- a **query** q — "what am I looking for?"
- a **key** k — "what do I contain / what can I be matched on?"
- a **value** v — "what information do I hand over if you attend to me?"

To build a new representation for token i:
1. **Score** every token j by how well its key matches i's query: the dot product qᵢ · kⱼ.
2. **Normalise** the scores into weights with softmax (positive, sum to 1).
3. **Mix**: output i = Σⱼ weightᵢⱼ · vⱼ.

For all tokens at once, with Q, K, V stacked as matrices (one row per token):

$$ Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V $$
:::where
Q — the queries, one row per token, shape (T, d_k)
K — the keys, shape (T, d_k)
V — the values, shape (T, d_v)
T — the number of tokens in the sequence
d_k — the length of each query and key vector
Q Kᵀ — the (T, T) table of every query · key dot product (Kᵀ is K transposed)
√d_k — the scale that stops the dot products from growing with the dimension
softmax(…) — applied to each row separately, turning one query's scores into weights that sum to 1
:::

QKᵀ is a (T × T) table of all query-key scores; softmax is applied to each row; multiplying by V produces one output vector per token.

:::key Why divide by √d_k?
Dot products of d_k-dimensional vectors with random-ish entries grow like √d_k. Large scores push softmax into a nearly one-hot, saturated regime where gradients vanish. Dividing by √d_k keeps the scores' scale steady regardless of dimension.
:::` },
      { type: 'demo', title: 'Interactive: attention weights', md: `Click a token to make it the query. The bars and the heat map show its attention distribution (each row sums to 1). Toggle the causal mask and change the score scale.`, demo: 'attention' },
      { type: 'text', title: 'Self-attention: Q, K and V come from the same sequence', md: `In **self-attention** all three come from the same input. Each token's embedding x is multiplied by three **learned** weight matrices:

$$ Q = X W_Q,     K = X W_K,     V = X W_V $$
:::where
X — the input sequence, one embedding per row, shape (T, d)
W_Q, W_K, W_V — three learned weight matrices, each of shape (d, d_k)
Q, K, V — the queries, keys and values, each (T, d_k): three different learned views of the same tokens
:::

The model *learns* what to look for (W_Q), how to be found (W_K) and what to pass along (W_V). Different trained heads learn different relationships — subject-verb links, coreference, "the previous token", matching brackets in code.

Shapes for a sequence of T tokens with model width d and head width d_k:

| | shape |
| --- | --- |
| X | (T, d) |
| W_Q, W_K, W_V | (d, d_k) |
| Q, K, V | (T, d_k) |
| Q Kᵀ (scores) | (T, T) |
| output | (T, d_k) |` },
      { type: 'code', title: 'Self-attention in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
tokens = ["The", "cat", "sat", "on", "the", "mat"]
T, d_model, d_k = len(tokens), 8, 4
X = rng.normal(size=(T, d_model))                      # token embeddings (random here)
Wq, Wk, Wv = (rng.normal(0, 0.5, (d_model, d_k)) for _ in range(3))

Q, K, V = X @ Wq, X @ Wk, X @ Wv                        # (T, d_k) each
scores = Q @ K.T / np.sqrt(d_k)                        # (T, T): how much each token looks at each other
mask = np.tril(np.ones((T, T), dtype=bool))            # causal: itself and earlier tokens only
scores = np.where(mask, scores, -np.inf)
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)          # softmax: each row sums to 1
out = weights @ V                                      # (T, d_k): weighted mix of value vectors

np.set_printoptions(precision=2, suppress=True)
print("attention weights (row = query, column = key):")
for t, row in zip(tokens, weights):
    print(f"{t:>4}", row)
print("output shape:", out.shape)
` },
      { type: 'check', question: { type: 'mcq', q: 'In softmax(QKᵀ/√d)V, what does row i of the softmax matrix represent?', options: ['The embedding of token i', 'How token i distributes its attention over all tokens (weights summing to 1)', 'The value vector of token i', 'The gradient for token i'], answer: 1, explain: 'Each row is a probability distribution over the keys for one query.' } },
      { type: 'text', title: 'The causal mask', md: `A language model predicts token t + 1 from tokens 1…t. During training we feed the whole sequence at once (that's what makes it parallel), so we must stop position t from peeking at later positions — otherwise it could simply copy the answer.

The **causal mask** sets the scores for all "future" positions (j > i) to −∞ before the softmax; e^(−∞) = 0, so they receive zero weight. The mask is lower-triangular:

\`\`\`text
        The  cat  sat
The   [  ✓    ·    · ]
cat   [  ✓    ✓    · ]
sat   [  ✓    ✓    ✓ ]
\`\`\`

GPT-style **decoder** models use causal self-attention. **Encoder** models like BERT use bidirectional (unmasked) attention because they encode a complete input rather than generate.` },
      { type: 'text', title: 'Cost: attention is quadratic', md: `The score matrix has T × T entries: doubling the context length quadruples attention's compute and memory. This is why context windows were long limited (512–4,096 tokens) and why a lot of research targets efficient attention: **FlashAttention** (exact attention computed in GPU-friendly blocks without materialising the full matrix), sparse and sliding-window attention, and linear-attention or state-space alternatives. During generation, the keys and values of previous tokens are cached (the **KV cache**) so each new token only computes its own query against them.` },
      { type: 'check', question: { type: 'tf', q: 'Without a causal mask, a model trained to predict the next token could cheat by attending to the very token it is supposed to predict.', answer: true, explain: 'The mask removes access to future positions during parallel training.' } }
    ],
    when: `- Any time elements of a set or sequence need to exchange information based on their content: text, code, image patches, audio frames, graph nodes, multimodal inputs.
- Causal self-attention for generation (decoders); bidirectional attention for understanding/encoding; **cross-attention** (queries from one sequence, keys/values from another) for translation, image captioning and diffusion conditioning.`,
    pitfalls: `- Forgetting the √d_k scaling (saturated softmax, poor training).
- Masking with 0 instead of −∞ before softmax (masked positions still get weight).
- Applying the mask after the softmax instead of before.
- Mixing up the matrix orientation: scores are Q Kᵀ (queries along rows).
- Underestimating memory for long contexts (T² growth).`,
    takeaways: [
      'Attention: each token mixes value vectors from other tokens, weighted by query-key similarity.',
      'Formula: softmax(QKᵀ/√d_k)V; each softmax row sums to 1.',
      'Self-attention learns W_Q, W_K, W_V to decide what to look for and what to pass on.',
      'The causal mask (−∞ above the diagonal) prevents peeking at future tokens in language models.',
      'Cost is O(T²) in sequence length; the KV cache speeds up generation.'
    ],
    quiz: [
      { type: 'mcq', q: 'In attention, what are the scores before the softmax?', options: ['The values', 'Dot products of queries with keys (scaled by 1/√d_k)', 'Distances between embeddings', 'Token ids'], answer: 1, explain: 'QKᵀ/√d_k measures how well each key matches each query.' },
      { type: 'fill', q: 'A sequence has 1,000 tokens. How many entries does one attention score matrix have?', answer: ['1000000', '1,000,000', '1e6', '10^6'], explain: 'T × T = 1,000 × 1,000 = 1,000,000.' },
      { type: 'mcq', q: 'What does the causal mask do?', options: ['Removes padding tokens', 'Prevents each position from attending to later positions', 'Scales the scores', 'Drops random attention weights'], answer: 1, explain: 'Future positions get −∞ scores → zero weight.' },
      { type: 'tf', q: 'In self-attention, the queries, keys and values are all computed from the same input sequence.', answer: true, explain: 'Q = XW_Q, K = XW_K, V = XW_V with the same X.' },
      { type: 'mcq', q: 'Why scale the scores by 1/√d_k?', options: ['To make attention faster', 'Large dot products saturate the softmax and kill gradients; scaling keeps them in a good range', 'To normalize the values', 'It is required for masking'], answer: 1, explain: 'Dot-product magnitudes grow with dimension.' }
    ],
    problems: [
      {
        id: 'causal-mask', title: 'The Causal Mask', difficulty: 'easy',
        prompt: `Implement \`causal_mask(T)\` → a boolean NumPy array of shape (T, T) where \`mask[i, j]\` is True when position i may attend to position j, i.e. when j ≤ i.

Then implement \`apply_mask(scores, mask)\` → a copy of \`scores\` with \`-np.inf\` wherever the mask is False.

\`\`\`text
causal_mask(3) -> [[ True, False, False],
                   [ True,  True, False],
                   [ True,  True,  True]]
\`\`\``,
        starter: `
import numpy as np

def causal_mask(T):
    # TODO
    pass

def apply_mask(scores, mask):
    # TODO
    pass
`,
        solution: `
import numpy as np

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))

def apply_mask(scores, mask):
    return np.where(mask, scores, -np.inf)
`,
        hints: ['`np.tril` keeps the lower triangle (including the diagonal).', '`np.where(mask, scores, -np.inf)` swaps in −inf where the mask is False.'],
        explain: `Lower-triangular means row i (a query) can see columns 0…i (itself and the past). Using −∞ (not a large negative number) guarantees exactly zero weight after the softmax.`,
        tests: [
          { call: 'causal_mask(3)', expect: 'np.array([[True, False, False], [True, True, False], [True, True, True]])' },
          { call: 'apply_mask(np.ones((2, 2)), causal_mask(2))', expect: 'np.array([[1.0, -np.inf], [1.0, 1.0]])' },
          { name: 'size 1', call: 'causal_mask(1)', expect: 'np.array([[True]])', hidden: true },
          { name: 'count of allowed pairs', code: 'm = causal_mask(10)\nassert m.dtype == bool and m.sum() == 55', hidden: true }
        ]
      },
      {
        id: 'scaled-dot-product-attention', title: 'Scaled Dot-Product Attention', difficulty: 'medium',
        prompt: `Implement \`attention(Q, K, V, mask=None)\` → \`(output, weights)\`:

1. scores = Q Kᵀ / √d_k, where d_k = Q.shape[-1]. Use \`np.swapaxes(K, -1, -2)\` so it also works with a leading batch/head dimension.
2. If a boolean \`mask\` is given (True = allowed), set disallowed scores to −∞.
3. weights = softmax over the **last** axis (numerically stable).
4. output = weights @ V.

\`\`\`text
If every key is identical, every query attends uniformly, so each output row is the mean of V.
\`\`\``,
        starter: `
import numpy as np

def attention(Q, K, V, mask=None):
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    # TODO
    pass
`,
        solution: `
import numpy as np

def attention(Q, K, V, mask=None):
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights
`,
        hints: ['Scores: `Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)`.', 'Subtract the row max before `np.exp` (−inf entries stay −inf and become 0).', 'Normalise along `axis=-1`.'],
        explain: `These few lines are the core computation of every transformer. Working on the last two axes (and using swapaxes instead of .T) means the same function handles one sequence, a batch, or a stack of heads.`,
        tests: [
          { name: 'identical keys give uniform attention', setup: 'Q = np.array([[1.0, 0.0], [0.0, 2.0]])\nK = np.ones((3, 2))\nV = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])', call: 'attention(Q, K, V)', expect: '(np.array([[1.0, 1.0], [1.0, 1.0]]), np.full((2, 3), 1 / 3))' },
          { name: 'causal mask', setup: 'Q = np.zeros((3, 2)); K = np.zeros((3, 2))\nV = np.array([[3.0], [6.0], [9.0]])\nmask = np.tril(np.ones((3, 3), dtype=bool))', call: 'attention(Q, K, V, mask)[0]', expect: 'np.array([[3.0], [4.5], [6.0]])' },
          { name: 'random inputs vs reference', setup: 'rng = np.random.default_rng(0)\nQ = rng.normal(size=(4, 3)); K = rng.normal(size=(5, 3)); V = rng.normal(size=(5, 2))', call: 'attention(Q, K, V)', hidden: true },
          { name: 'batched heads with mask', setup: 'rng = np.random.default_rng(1)\nQ = rng.normal(size=(2, 4, 3)); K = rng.normal(size=(2, 4, 3)); V = rng.normal(size=(2, 4, 5))\nmask = np.tril(np.ones((4, 4), dtype=bool))', call: 'attention(Q, K, V, mask)', hidden: true },
          { name: 'rows of weights sum to 1', setup: 'rng = np.random.default_rng(2)\nQ = rng.normal(size=(6, 4)) * 10; K = rng.normal(size=(6, 4)) * 10; V = rng.normal(size=(6, 4))', code: 'out, w = attention(Q, K, V)\nassert np.allclose(w.sum(axis=-1), 1)\nassert np.all(np.isfinite(out))', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Vaswani et al. — Attention Is All You Need (2017)', url: 'https://arxiv.org/abs/1706.03762' },
      { title: 'Jay Alammar — The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' },
      { title: '3Blue1Brown — Attention in transformers, visually explained', url: 'https://www.3blue1brown.com/lessons/attention' }
    ]
  },

  // =====================================================================================
  {
    id: 'multi-head-attention', branch: 'llm', title: 'Multi-Head Attention', level: 3, minutes: 40, prereqs: ['attention'],
    tags: ['multi-head attention', 'heads', 'projection', 'reshape', 'gqa', 'mqa'],
    summary: 'Running several attention operations in parallel so each head can track a different kind of relationship: splitting into heads, shapes, the output projection, and GQA for efficient inference.',
    why: `A single attention head computes one weighting over the sequence — but a token usually needs several kinds of information at once: its syntactic head, the entity it refers to, the previous token, the matching bracket. **Multi-head attention** runs many smaller attention operations in parallel, each free to specialise. It is the exact form used in every transformer layer, and understanding its reshapes and parameter counts is essential for reading model code and reasoning about inference memory (the KV cache), where variants like grouped-query attention matter.`,
    objectives: [
      'Explain why multiple heads help',
      'Split a (T, d_model) tensor into h heads of size d_model/h and merge them back',
      'Implement multi-head attention with Q/K/V projections and the output projection W_O',
      'Count its parameters and explain multi-query / grouped-query attention'
    ],
    sections: [
      { type: 'text', title: 'Several kinds of attention at once', md: `Multi-head attention runs h attention "heads" in parallel. Each head has its own (smaller) query, key and value projections, so it can learn its own notion of relevance. Interpretability research finds heads that, for example:
- attend to the **previous token**;
- link a pronoun to its **antecedent**;
- connect a verb with its **subject or object**;
- implement "**induction**": find an earlier occurrence of the current token and copy what followed it — a key mechanism behind in-context learning.

The heads' outputs are concatenated and mixed by a final linear layer W_O.` },
      { type: 'text', title: 'The mechanics and shapes', md: `With model width d_model and h heads, each head has width d_head = d_model / h (e.g. 768 / 12 = 64 in GPT-2 small).

1. Project: Q = X W_Q, K = X W_K, V = X W_V, each (T, d_model).
2. **Split** each into heads: reshape (T, d_model) → (T, h, d_head), then transpose to **(h, T, d_head)**.
3. Run scaled dot-product attention **independently per head** (scores (h, T, T)) — one batched matrix multiply.
4. **Merge**: transpose back to (T, h, d_head) and reshape to (T, d_model) — concatenating the heads.
5. Output projection: out = merged @ W_O, (T, d_model).

Splitting into heads costs nothing extra: the total compute is about the same as one big head of width d_model, but the model gets h independent attention patterns.

**Parameters:** W_Q, W_K, W_V, W_O are each d_model × d_model → 4·d_model² (plus biases if used). For GPT-2 small: 4 × 768² ≈ 2.36M per layer.` },
      { type: 'code', title: 'Multi-head attention in NumPy', code: `
import numpy as np
rng = np.random.default_rng(1)
T, d_model, n_heads = 5, 8, 2
d_head = d_model // n_heads
X = rng.normal(size=(T, d_model))
Wq, Wk, Wv, Wo = (rng.normal(0, 0.3, (d_model, d_model)) for _ in range(4))

def split(M):                                   # (T, d_model) -> (n_heads, T, d_head)
    return M.reshape(T, n_heads, d_head).transpose(1, 0, 2)

Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)     # (n_heads, T, T)
mask = np.tril(np.ones((T, T), dtype=bool))
scores = np.where(mask, scores, -np.inf)
w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
heads = w @ V                                           # (n_heads, T, d_head)
merged = heads.transpose(1, 0, 2).reshape(T, d_model)   # concatenate heads
out = merged @ Wo                                       # (T, d_model)

print("each head's attention for the last token:")
print(w[:, -1].round(2))
print("output shape:", out.shape, "  attention parameters:", 4 * d_model * d_model)
` },
      { type: 'check', question: { type: 'fill', q: 'A model has d_model = 1024 and 16 heads. What is the dimension of each head?', answer: ['64'], explain: '1024 / 16 = 64.' } },
      { type: 'text', title: 'Multi-query and grouped-query attention', md: `During generation, a model caches the keys and values of every previous token for every head and layer (the **KV cache**). For long contexts and big batches this cache — not the weights — dominates GPU memory.

- **Multi-query attention (MQA)**: all heads share **one** key head and one value head (queries stay per-head). The KV cache shrinks by a factor of h, at a small quality cost.
- **Grouped-query attention (GQA)**: heads are divided into g groups that each share a key/value head — a middle ground (e.g. 32 query heads with 8 KV heads). Used by Llama 2 70B, Llama 3, Mistral and many others.

The math per head is unchanged; only the number of distinct K/V projections differs.` },
      { type: 'check', question: { type: 'tf', q: 'Splitting attention into 12 heads multiplies the attention computation cost by roughly 12.', answer: false, explain: 'Each head has width d_model/12, so total work stays about the same as one full-width head.' } },
      { type: 'text', title: 'Going further: what heads learn and what they cost', md: `- **Specialization**: trained heads attend to different things — the previous token, syntactic relations, matching brackets, the first token of the sequence (an "attention sink"). Many heads are redundant and can be pruned with little loss.
- **Cost**: the projections W_Q, W_K, W_V and W_O hold about 4 · d_model² parameters per layer whatever the number of heads. Splitting d_model into h heads of size d_model / h keeps the compute the same while letting each head learn its own pattern.
- **Smaller KV caches**: multi-query attention (one shared key/value head) and grouped-query attention (one key/value head per group) shrink the KV cache (see *Efficient Attention*); multi-head latent attention compresses keys and values into a small latent vector.
- **Reading attention maps**: attention weights show where information flows, but they are an incomplete explanation — heads work together in circuits such as induction heads (see *Mechanistic Interpretability*).` }
    ],
    when: `- Multi-head attention is the default in every transformer layer.
- Use GQA/MQA variants when inference memory (KV cache) and speed matter — most modern open LLMs do.
- Keep d_model divisible by the number of heads; typical head sizes are 64–128.`,
    pitfalls: `- Reshape vs transpose confusion: reshape (T, d) → (T, h, d_head) **then** transpose to (h, T, d_head). Reshaping directly to (h, T, d_head) scrambles the data.
- Forgetting the output projection W_O.
- Scaling by √d_model instead of √d_head.
- Mixing up which axis is heads when merging back.`,
    takeaways: [
      'Multi-head attention runs h attention operations in parallel, each with width d_model/h.',
      'Split: reshape (T, d) → (T, h, d_head) → transpose (h, T, d_head); merge reverses it.',
      'Heads specialise (previous token, coreference, induction…).',
      'Parameters: W_Q, W_K, W_V, W_O → 4·d_model².',
      'MQA/GQA share key/value heads to shrink the KV cache for fast inference.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the shape of the per-head score tensor for h heads and T tokens?', options: ['(T, d_model)', '(h, T, T)', '(T, h)', '(d_head, d_head)'], answer: 1, explain: 'Each head has its own T × T score matrix.' },
      { type: 'fill', q: 'How many parameters (no biases) are in the Q, K, V and output projections of one attention layer with d_model = 512?', answer: ['1048576', '1,048,576', '4*512^2'], explain: '4 × 512² = 1,048,576.' },
      { type: 'mcq', q: 'What is the main benefit of grouped-query attention?', options: ['Higher accuracy than full attention', 'A smaller KV cache and faster inference by sharing key/value heads', 'No need for positional information', 'Removes the softmax'], answer: 1, explain: 'Fewer distinct K/V heads means less memory per cached token.' },
      { type: 'tf', q: 'The outputs of all heads are concatenated and then multiplied by an output projection matrix W_O.', answer: true, explain: 'W_O mixes information across heads.' },
      { type: 'mcq', q: 'An "induction head" is believed to help with…', options: ['tokenization', 'in-context learning by copying what followed earlier occurrences of the current token', 'positional encoding', 'weight decay'], answer: 1, explain: 'It looks back for the previous occurrence of a pattern and predicts its continuation.' }
    ],
    problems: [
      {
        id: 'split-merge-heads', title: 'Split and Merge Heads', difficulty: 'medium',
        prompt: `Implement the reshapes used by multi-head attention:

- \`split_heads(X, n_heads)\`: X has shape (T, d_model) → return shape (n_heads, T, d_model // n_heads), where head h holds columns h·d_head … (h+1)·d_head − 1 of X. (Reshape to (T, n_heads, d_head), then transpose.)
- \`merge_heads(H)\`: the exact inverse, (n_heads, T, d_head) → (T, n_heads·d_head).

\`\`\`text
X = np.arange(12).reshape(2, 6);  split_heads(X, 3)[1]  ->  [[2, 3], [8, 9]]
\`\`\``,
        starter: `
import numpy as np

def split_heads(X, n_heads):
    # TODO
    pass

def merge_heads(H):
    # TODO
    pass
`,
        solution: `
import numpy as np

def split_heads(X, n_heads):
    X = np.asarray(X)
    T, d = X.shape
    return X.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)

def merge_heads(H):
    H = np.asarray(H)
    h, T, dh = H.shape
    return H.transpose(1, 0, 2).reshape(T, h * dh)
`,
        hints: ['Reshape first: (T, n_heads, d_head) keeps each token\'s columns together.', 'Then `.transpose(1, 0, 2)` moves the head axis to the front.', 'Merging: transpose back, then reshape.'],
        explain: `Reshaping (T, d) to (T, h, d_head) chops each token's vector into consecutive chunks — one per head — without moving data; the transpose brings heads to the front so attention runs as one batched matmul. Reshaping straight to (h, T, d_head) would mix different tokens into one head: a classic bug.`,
        tests: [
          { call: 'split_heads(np.arange(12).reshape(2, 6), 3)[1]', expect: 'np.array([[2, 3], [8, 9]])' },
          { name: 'shape', code: 'H = split_heads(np.zeros((7, 12)), 4)\nassert H.shape == (4, 7, 3), H.shape' },
          { name: 'round trip', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(5, 8))', code: 'assert np.allclose(merge_heads(split_heads(X, 2)), X)\nassert np.allclose(merge_heads(split_heads(X, 4)), X)', hidden: true },
          { name: 'random split', setup: 'rng = np.random.default_rng(3)\nX = rng.normal(size=(3, 6))', call: 'split_heads(X, 3)', hidden: true }
        ]
      },
      {
        id: 'mha-forward', title: 'Multi-Head Attention Forward Pass', difficulty: 'hard',
        prompt: `Implement \`mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True)\`:

1. Q, K, V = X @ Wq, X @ Wk, X @ Wv (all (T, d_model)).
2. Split each into heads (n_heads, T, d_head) — \`split_heads\` and \`merge_heads\` are provided.
3. Per head: scores = Q Kᵀ / √d_head; if \`causal\`, mask future positions with −∞; stable softmax over the last axis; heads = weights @ V.
4. Merge the heads back to (T, d_model) and return \`merged @ Wo\`.`,
        starter: `
import numpy as np

def split_heads(X, n_heads):
    T, d = X.shape
    return X.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)

def merge_heads(H):
    h, T, dh = H.shape
    return H.transpose(1, 0, 2).reshape(T, h * dh)

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    X = np.asarray(X, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def split_heads(X, n_heads):
    T, d = X.shape
    return X.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)

def merge_heads(H):
    h, T, dh = H.shape
    return H.transpose(1, 0, 2).reshape(T, h * dh)

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    d_head = d // n_heads
    Q, K, V = split_heads(X @ Wq, n_heads), split_heads(X @ Wk, n_heads), split_heads(X @ Wv, n_heads)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)
    if causal:
        scores = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return merge_heads(w @ V) @ Wo
`,
        hints: ['After splitting, `Q @ K.transpose(0, 2, 1)` gives (n_heads, T, T).', 'The (T, T) causal mask broadcasts across heads.', 'Scale by √d_head, not √d_model.'],
        explain: `Everything per head happens in a single batched matrix multiply, which is why multi-head attention is no slower than single-head attention of the same total width. The hidden causality test checks that changing a later token never changes earlier outputs.`,
        tests: [
          { name: 'random weights vs reference', setup: 'rng = np.random.default_rng(0)\nT, d = 5, 8\nX = rng.normal(size=(T, d))\nWq, Wk, Wv, Wo = (rng.normal(0, 0.3, (d, d)) for _ in range(4))', call: 'mha(X, Wq, Wk, Wv, Wo, 2)' },
          { name: 'shape preserved', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(6, 12))\nW = [rng.normal(0, 0.2, (12, 12)) for _ in range(4)]', code: 'out = mha(X, *W, 3)\nassert out.shape == (6, 12)' },
          { name: 'causality: the future cannot affect the past', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(6, 8))\nW = [rng.normal(0, 0.3, (8, 8)) for _ in range(4)]', code: 'a = mha(X, *W, 4)\nX2 = X.copy(); X2[4:] = rng.normal(size=(2, 8))\nb = mha(X2, *W, 4)\nassert np.allclose(a[:4], b[:4]), "outputs for positions 0..3 must not depend on later tokens"\nassert not np.allclose(a[4:], b[4:])', hidden: true },
          { name: 'non-causal version', setup: 'rng = np.random.default_rng(4)\nX = rng.normal(size=(4, 6))\nW = [rng.normal(0, 0.3, (6, 6)) for _ in range(4)]', call: 'mha(X, *W, 3, causal=False)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Anthropic — A Mathematical Framework for Transformer Circuits', url: 'https://transformer-circuits.pub/2021/framework/index.html', note: 'how heads compose (induction heads)' },
      { title: 'Ainslie et al. — GQA: Grouped-Query Attention', url: 'https://arxiv.org/abs/2305.13245' }
    ]
  },

  // =====================================================================================
  {
    id: 'positional-encoding', branch: 'llm', title: 'Positional Encoding', level: 2, minutes: 35, prereqs: ['attention'],
    tags: ['positional encoding', 'sinusoidal', 'learned positions', 'rope', 'rotary', 'alibi', 'context length'],
    summary: 'Attention has no built-in notion of order. How transformers inject position: learned position embeddings, sinusoidal encodings, and rotary embeddings (RoPE) used by modern LLMs.',
    why: `Self-attention treats its input as a **set**: shuffle the tokens and each output is simply shuffled along — "dog bites man" and "man bites dog" would get identical treatment. Order matters enormously in language, so transformers must add positional information. How they do it affects how well a model generalizes to longer contexts; **RoPE** (rotary position embedding), used by Llama, Mistral and most recent open LLMs, is a big part of how context windows have been stretched to hundreds of thousands of tokens.`,
    objectives: [
      'Show that self-attention without positions is permutation-equivariant',
      'Compute sinusoidal positional encodings and explain their properties',
      'Compare learned absolute, sinusoidal, relative and rotary (RoPE) schemes',
      'Explain how RoPE makes attention depend on relative position'
    ],
    sections: [
      { type: 'text', title: 'Attention is order-blind', md: `Attention computes weights from dot products between token vectors. Nothing in softmax(QKᵀ/√d)V depends on *where* a token sits: permute the input rows and the output rows are permuted the same way (**permutation equivariance**). The model would see a bag of tokens.

The fix is to make each token's vector depend on its position, either by **adding** a position vector to the token embedding before the first layer (absolute positions), or by modifying how queries and keys interact (relative positions, RoPE, ALiBi).` },
      { type: 'code', title: 'Proving attention ignores order — and fixing it', code: `
import numpy as np
rng = np.random.default_rng(0)
T, d = 4, 6
X = rng.normal(size=(T, d)); Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))

def self_attention(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    s = Q @ K.T / np.sqrt(d)
    w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return w @ V

perm = [2, 0, 3, 1]
print("no positions: shuffled input -> just shuffled output:", np.allclose(self_attention(X)[perm], self_attention(X[perm])))

def sinusoidal(T, d):
    pos = np.arange(T)[:, None]; i = np.arange(0, d, 2)[None, :]
    angles = pos / 10000 ** (i / d)
    pe = np.zeros((T, d)); pe[:, 0::2] = np.sin(angles); pe[:, 1::2] = np.cos(angles)
    return pe

P = sinusoidal(T, d)
print("with positions added: still just a shuffle?         ", np.allclose(self_attention(X + P)[perm], self_attention(X[perm] + P)))
print("the first 4 positional vectors:")
print(P.round(3))
` },
      { type: 'text', title: 'Sinusoidal encodings', md: `The original Transformer added a fixed vector to each position:

$$ PE(pos, 2i) = sin( pos / 10000^(2i/d) )          PE(pos, 2i+1) = cos( pos / 10000^(2i/d) ) $$
:::where
pos — the token's position in the sequence (0, 1, 2, …)
d — the model width, which is also the length of each positional vector
i — the index of a pair of dimensions, from 0 to d/2 − 1; dimensions 2i and 2i + 1 share one frequency
PE(pos, j) — dimension j of the encoding vector for position pos
10000^(2i/d) — the wavelength factor: 1 for the first pair, nearly 10,000 for the last, so the waves go from fast to slow
:::

Each pair of dimensions is a sine/cosine wave with its own frequency — fast oscillations in the first dimensions, very slow ones in the last, like the hands of a clock (seconds, minutes, hours…). Properties:
- every position gets a unique pattern;
- the encoding for position p + k is a fixed **rotation** of the encoding for p, so relative offsets are easy for the model to compute;
- no parameters to learn, and it is defined for any position (though models don't automatically generalize past the lengths they were trained on).` },
      { type: 'demo', title: 'Interactive: the sinusoidal encoding matrix', demo: 'positional' },
      { type: 'text', title: 'Learned, relative and rotary positions', md: `- **Learned absolute embeddings** (BERT, GPT-2): a trainable vector per position, added to the token embedding. Simple, works well, but can't go beyond the maximum position it was trained with.
- **Relative position biases** (Transformer-XL, T5): add a learned bias to the attention score depending on the distance j − i.
- **ALiBi**: subtract a penalty proportional to distance from each attention score — no embeddings at all; extrapolates well.
- **RoPE (rotary position embedding)** (Llama, Mistral, Qwen, most current open models): instead of adding anything to the input, **rotate** each query and key vector by an angle proportional to its position. Treat dimensions in pairs (x₂ᵢ, x₂ᵢ₊₁) as 2-D points and rotate pair i by pos·θᵢ with θᵢ = 10000^(−2i/d):

$$ (x₂ᵢ, x₂ᵢ₊₁) → (x₂ᵢ cos φ − x₂ᵢ₊₁ sin φ,   x₂ᵢ sin φ + x₂ᵢ₊₁ cos φ),   φ = pos · θᵢ $$
:::where
x — a query or key vector of length d
(x₂ᵢ, x₂ᵢ₊₁) — its i-th pair of dimensions, treated as a point in a 2-D plane
pos — the position of the token
θᵢ — "theta i": the rotation speed of pair i, 10000^(−2i/d), fast for the first pairs and slow for the last
φ — "phi": the rotation angle, pos · θᵢ
cos φ, sin φ — the standard 2-D rotation by the angle φ
:::

Because rotating q by angle mθ and k by angle nθ changes their dot product only through the **difference** (m − n)θ, the attention score depends on the relative distance between tokens — relative positioning, implemented with absolute rotations. Context-extension tricks (position interpolation, NTK scaling, YaRN) adjust these rotation frequencies so models can handle longer contexts than they were trained on.` },
      { type: 'check', question: { type: 'mcq', q: 'Why does RoPE make attention scores depend on relative position?', options: ['It adds the distance to every score', 'Rotating q by mθ and k by nθ leaves their dot product depending only on (m − n)θ', 'It sorts the tokens', 'It uses learned position embeddings'], answer: 1, explain: 'Rotations preserve dot products up to the angle difference.' } }
    ],
    when: `- Any transformer needs positional information; use what the model family uses when fine-tuning (you can't swap schemes casually).
- RoPE is the modern default for decoder LLMs; learned absolute embeddings remain fine for fixed-length encoders; ALiBi/RoPE variants when length extrapolation matters.`,
    pitfalls: `- Forgetting positions entirely (bag-of-tokens model).
- Running a model on sequences longer than it was trained for and expecting good quality without context-extension techniques.
- Applying RoPE to the values (it is applied to queries and keys only).
- Mixing up dimension pairing conventions between implementations (interleaved vs half-split) when porting weights.`,
    takeaways: [
      'Self-attention alone is order-blind (permutation-equivariant).',
      'Absolute schemes add a position vector to each token embedding (learned or sinusoidal).',
      'Sinusoidal encodings use sine/cosine waves of different frequencies per dimension pair.',
      'RoPE rotates query/key pairs by position-dependent angles, so scores depend on relative distance.',
      'Context extension tweaks RoPE frequencies; models don\'t automatically generalize beyond trained lengths.'
    ],
    quiz: [
      { type: 'tf', q: 'Without positional information, a transformer\'s self-attention output for "dog bites man" is just a reordering of its output for "man bites dog".', answer: true, explain: 'Attention is permutation-equivariant; positions are needed to distinguish order.' },
      { type: 'mcq', q: 'In sinusoidal encodings, the low-index dimensions…', options: ['oscillate slowly', 'oscillate quickly (high frequency)', 'are always zero', 'are learned'], answer: 1, explain: 'The frequency decreases with the dimension index: early dims are the "seconds hand".' },
      { type: 'mcq', q: 'Which positional method do Llama-family models use?', options: ['Learned absolute embeddings', 'Rotary position embeddings (RoPE)', 'No positional information', 'One-hot positions'], answer: 1, explain: 'RoPE is applied to queries and keys in every attention layer.' },
      { type: 'mcq', q: 'RoPE is applied to…', options: ['the values', 'the queries and keys', 'the output projection', 'the token ids'], answer: 1, explain: 'Rotating q and k shapes the attention scores; values are left alone.' },
      { type: 'fill', q: 'For the sinusoidal encoding at position 0, what is the value of every sine dimension?', answer: ['0', '0.0'], explain: 'sin(0) = 0 (and every cosine dimension is 1).' }
    ],
    problems: [
      {
        id: 'sinusoidal-pe', title: 'Sinusoidal Positional Encoding', difficulty: 'medium',
        prompt: `Implement \`sinusoidal_pe(T, d)\` (d even) returning a (T, d) array with

\`\`\`text
PE[pos, 2i]   = sin(pos / 10000 ** (2i / d))
PE[pos, 2i+1] = cos(pos / 10000 ** (2i / d))
\`\`\`

for pos = 0 … T−1 and i = 0 … d/2 − 1. Try to vectorize: build a column of positions and a row of frequencies and let broadcasting form the angle matrix.

\`\`\`text
sinusoidal_pe(2, 2)  ->  [[0., 1.], [sin(1), cos(1)]]
\`\`\``,
        starter: `
import numpy as np

def sinusoidal_pe(T, d):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]                 # (T, 1)
    two_i = np.arange(0, d, 2)[None, :]         # (1, d/2): values 0, 2, 4, ...
    angles = pos / 10000 ** (two_i / d)         # (T, d/2)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
`,
        hints: ['`np.arange(0, d, 2)` gives the values 2i directly.', '`pe[:, 0::2]` addresses the even columns, `pe[:, 1::2]` the odd ones.'],
        explain: `Broadcasting a (T, 1) column of positions against a (1, d/2) row of frequencies builds every angle at once. Interleaving sin and cos per frequency gives each pair of dimensions the rotation structure that makes relative offsets linear functions of the encodings.`,
        tests: [
          { call: 'sinusoidal_pe(2, 2)', expect: 'np.array([[0.0, 1.0], [np.sin(1.0), np.cos(1.0)]])' },
          { name: 'second frequency', call: 'sinusoidal_pe(3, 4)[2]', expect: 'np.array([np.sin(2.0), np.cos(2.0), np.sin(2 / 100), np.cos(2 / 100)])' },
          { name: 'position 0', code: 'P = sinusoidal_pe(5, 8)\nassert np.allclose(P[0, 0::2], 0) and np.allclose(P[0, 1::2], 1)', hidden: true },
          { name: 'larger table', call: 'sinusoidal_pe(10, 6)', hidden: true }
        ]
      },
      {
        id: 'rope-rotate', title: 'Rotary Position Embedding (RoPE)', difficulty: 'hard',
        prompt: `Implement \`apply_rope(x)\` for x of shape (T, d) with d even. Row t is the query (or key) at position t.

For each pair index i = 0 … d/2 − 1 with frequency θᵢ = 10000^(−2i/d) and angle φ = t·θᵢ, rotate the pair of **adjacent** dimensions (2i, 2i+1):

\`\`\`text
out[t, 2i]   = x[t, 2i] * cos(φ) - x[t, 2i+1] * sin(φ)
out[t, 2i+1] = x[t, 2i] * sin(φ) + x[t, 2i+1] * cos(φ)
\`\`\`

The tests check that rotation preserves each vector's length and that the dot product between a rotated query at position m and a rotated key at position n depends only on m − n.`,
        starter: `
import numpy as np

def apply_rope(x):
    x = np.asarray(x, dtype=float)
    T, d = x.shape
    # TODO
    pass
`,
        solution: `
import numpy as np

def apply_rope(x):
    x = np.asarray(x, dtype=float)
    T, d = x.shape
    theta = 10000.0 ** (-np.arange(0, d, 2) / d)          # (d/2,)
    phi = np.arange(T)[:, None] * theta[None, :]          # (T, d/2)
    cos, sin = np.cos(phi), np.sin(phi)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out
`,
        hints: ['θ for pair i is `10000 ** (-(2i) / d)`; `np.arange(0, d, 2)` gives 2i.', 'Build the (T, d/2) angle table with broadcasting.', 'Even columns `x[:, 0::2]` and odd columns `x[:, 1::2]` form the pairs.'],
        explain: `Each pair is a 2-D point rotated by an angle that grows with position. Rotations keep lengths unchanged, and (R_m q)·(R_n k) = q·(R_{n−m} k): the score only sees the relative offset. That is the whole trick — implemented in a few vectorized lines inside every Llama layer.`,
        tests: [
          { name: 'position 0 is unchanged', call: 'apply_rope(np.array([[1.0, 2.0, 3.0, 4.0]]))', expect: 'np.array([[1.0, 2.0, 3.0, 4.0]])' },
          { name: 'single pair rotates by t radians', call: 'apply_rope(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))', expect: 'np.array([[1.0, 0.0], [np.cos(1.0), np.sin(1.0)], [np.cos(2.0), np.sin(2.0)]])' },
          { name: 'lengths preserved', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=(9, 8))', code: 'r = apply_rope(x)\nassert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(x, axis=1))', hidden: true },
          { name: 'scores depend only on relative position', setup: 'rng = np.random.default_rng(1)\nq = rng.normal(size=8); k = rng.normal(size=8)\nRQ = apply_rope(np.tile(q, (12, 1))); RK = apply_rope(np.tile(k, (12, 1)))', code: 'assert np.isclose(RQ[5] @ RK[2], RQ[9] @ RK[6]), "offset 3 should give the same score anywhere"\nassert np.isclose(RQ[4] @ RK[4], q @ k), "same position: plain dot product"\nassert not np.isclose(RQ[5] @ RK[2], RQ[5] @ RK[4])', hidden: true },
          { name: 'random matrix vs reference', setup: 'rng = np.random.default_rng(2)\nx = rng.normal(size=(5, 6))', call: 'apply_rope(x)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding', url: 'https://arxiv.org/abs/2104.09864' },
      { title: 'Press et al. — Train Short, Test Long (ALiBi)', url: 'https://arxiv.org/abs/2108.12409' }
    ]
  },

  // =====================================================================================
  {
    id: 'transformer-block', branch: 'llm', title: 'The Transformer Block', level: 3, minutes: 45, prereqs: ['multi-head-attention', 'positional-encoding', 'normalization'],
    tags: ['transformer', 'residual stream', 'pre-norm', 'feed-forward', 'encoder', 'decoder'],
    summary: 'Assembling the pieces: pre-norm residual blocks of attention plus a feed-forward MLP, the "residual stream", stacking N blocks, and encoder vs decoder vs encoder-decoder designs.',
    why: `The transformer block is the repeating unit of every modern language model: GPT-3 stacks 96 of them, Llama 3 8B stacks 32. Once you understand one block — normalize, attend, add; normalize, MLP, add — you understand the architecture behind ChatGPT, Claude, Gemini, BERT, vision transformers and much of image generation. Everything else (bigger widths, more layers, better data) is scale.`,
    objectives: [
      'Write a pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))',
      'Explain the roles of attention (mixing across tokens) and the FFN (processing each token)',
      'Describe the residual stream view and why shapes stay (T, d_model) throughout',
      'Compare encoder-only, decoder-only and encoder-decoder transformers, and count block parameters'
    ],
    sections: [
      { type: 'text', title: 'One block', md: `A modern (pre-norm) decoder block maps a (T, d_model) matrix to another (T, d_model) matrix:

\`\`\`text
x = x + MultiHeadAttention( LayerNorm(x) )     # tokens exchange information
x = x + FeedForward( LayerNorm(x) )            # each token is processed independently
\`\`\`

- **Attention** is the only place where information moves **between** positions.
- The **feed-forward network** (FFN, an MLP) is applied to every position separately with the same weights: typically Linear(d → 4d) → GELU (or SwiGLU) → Linear(4d → d). It holds most of the block's parameters and is thought to store much of the model's factual knowledge.
- **LayerNorm/RMSNorm** before each sub-layer keeps activations well-scaled (*pre-norm* trains more stably than the original post-norm).
- **Residual connections** (the "x +") let each sub-layer add a small update and give gradients a highway through dozens of layers.` },
      { type: 'demo', title: 'Diagram: the block at a glance', demo: 'dg-transformer' },
      { type: 'text', title: 'The residual stream', md: `A helpful mental model: each token has a vector of width d_model — the **residual stream** — that flows straight through the network from the embedding to the output. Every attention and FFN sub-layer **reads** from the stream (after normalization) and **writes** an update back by addition. Nothing overwrites the stream; layers only add to it.

That's why shapes never change inside the stack: input (T, d_model) → after block 1 (T, d_model) → … → after block N (T, d_model). At the end, a final norm and the output layer turn each position's vector into logits over the vocabulary.

\`\`\`text
tokens → embedding (+ positions) → [block] × N → final norm → linear → logits (T, vocab)
\`\`\`` },
      { type: 'code', title: 'A transformer block in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
T, d, d_ff = 5, 16, 64

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def causal_self_attention(x, Wq, Wk, Wv, Wo):        # single head for brevity
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    s = Q @ K.T / np.sqrt(x.shape[1])
    s = np.where(np.tril(np.ones((len(x), len(x)), bool)), s, -np.inf)
    w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return (w @ V) @ Wo

P = {k: rng.normal(0, 0.1, (d, d)) for k in ["Wq", "Wk", "Wv", "Wo"]}
W1, W2 = rng.normal(0, 0.1, (d, d_ff)), rng.normal(0, 0.1, (d_ff, d))

def block(x):
    x = x + causal_self_attention(layer_norm(x), P["Wq"], P["Wk"], P["Wv"], P["Wo"])   # mix across tokens
    x = x + gelu(layer_norm(x) @ W1) @ W2                                                # per-token MLP
    return x

x = rng.normal(size=(T, d))
y = x
for _ in range(4):            # stacking (real models use different weights in every block)
    y = block(y)
print("shape in:", x.shape, " shape out:", y.shape)
print("parameters in one block (attention 4d^2 + FFN 2*d*4d):", 4 * d * d + 2 * d * d_ff)
` },
      { type: 'check', question: { type: 'mcq', q: 'In a transformer block, where does information move between different token positions?', options: ['In the feed-forward network', 'Only in the attention sub-layer', 'In the LayerNorm', 'In the residual addition'], answer: 1, explain: 'The FFN and normalization act on each position independently; attention mixes across positions.' } },
      { type: 'text', title: 'Encoders, decoders and encoder-decoders', md: `| Type | Attention | Trained to | Examples | Good for |
| --- | --- | --- | --- | --- |
| **Encoder-only** | bidirectional (no mask) | fill in masked tokens | BERT, RoBERTa | classification, embeddings, search |
| **Decoder-only** | causal (masked) | predict the next token | GPT, Llama, Claude | generation, chat, general-purpose LLMs |
| **Encoder-decoder** | encoder bidirectional; decoder causal + **cross-attention** to the encoder | map input to output text | original Transformer, T5, Whisper | translation, summarization, speech-to-text |

**Cross-attention** takes queries from the decoder and keys/values from the encoder's output — how the decoder "reads" the source sentence. Decoder-only models dominate today because next-token prediction on raw text scales so well and handles every task as text completion.` },
      { type: 'text', title: 'Counting parameters', md: `Per block (ignoring biases and the small norm parameters), with d = d_model and FFN width 4d:

$$ attention: 4d²          FFN: 2 · d · 4d = 8d²          total ≈ 12 d² $$
:::where
d — the model width, d_model
4d² — the four d × d attention matrices: W_Q, W_K, W_V and W_O
2 · d · 4d — the FFN's two matrices, d × 4d (up) and 4d × d (down)
total ≈ 12d² — the parameters of one block (biases and norms add only about 13d more)
:::

GPT-2 small: d = 768, 12 blocks → 12 × 12 × 768² ≈ 85M parameters in the blocks, plus the 50,257 × 768 ≈ 38.6M token-embedding matrix ≈ 124M total. GPT-3: d = 12,288, 96 blocks → 96 × 12 × 12,288² ≈ 174B — essentially its famous 175B. (Models with SwiGLU FFNs use three FFN matrices with a ~8d/3 hidden width to keep the count similar.)` },
      { type: 'check', question: { type: 'fill', q: 'Using ≈12·d² parameters per block, how many parameters (in millions, to the nearest million) are in one block with d = 1024?', answer: ['13', '12.6', '12.58', '12.583'], display: '≈ 12.6M', explain: '12 × 1024² = 12,582,912 ≈ 12.6M.' } }
    ],
    when: `- Decoder-only transformers for generative LLMs and most general-purpose language tasks.
- Encoder-only models for classification, retrieval embeddings and token tagging when generation isn't needed (smaller and faster).
- Encoder-decoder models for sequence-to-sequence tasks with a distinct input and output (translation, speech recognition).`,
    pitfalls: `- Forgetting the residual additions (the network becomes much harder to train).
- Post-norm vs pre-norm mix-ups when porting code (pre-norm is the modern default).
- Assuming the FFN mixes tokens — it doesn't; only attention does.
- Mis-sizing the FFN hidden width (4·d, or ≈ 8d/3 with SwiGLU's three matrices).`,
    takeaways: [
      'Pre-norm block: x ← x + Attn(LN(x)); x ← x + FFN(LN(x)).',
      'Attention mixes information across positions; the FFN processes each position independently.',
      'The residual stream keeps shape (T, d_model) through all N blocks; layers add updates to it.',
      'Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) differ in masking and cross-attention.',
      'Parameters per block ≈ 12·d_model².'
    ],
    quiz: [
      { type: 'order', q: 'Order the operations in a pre-norm transformer block.', items: ['LayerNorm the input', 'Multi-head self-attention', 'Add to the residual stream', 'LayerNorm again', 'Feed-forward network', 'Add to the residual stream'], explain: 'Norm → attention → add; norm → FFN → add.' },
      { type: 'mcq', q: 'What is the typical hidden width of a transformer FFN relative to d_model?', options: ['d_model / 4', 'd_model', '4 × d_model', 'vocabulary size'], answer: 2, explain: 'Classically 4×d_model (≈ 8/3×d_model with SwiGLU\'s three matrices).' },
      { type: 'mcq', q: 'Which architecture is GPT?', options: ['Encoder-only', 'Decoder-only', 'Encoder-decoder', 'Convolutional'], answer: 1, explain: 'A stack of causal self-attention blocks trained on next-token prediction.' },
      { type: 'tf', q: 'Cross-attention uses queries from one sequence and keys/values from another.', answer: true, explain: 'E.g. the decoder\'s queries attend over the encoder\'s outputs.' },
      { type: 'mcq', q: 'What does the residual stream view say?', options: ['Each layer replaces the token vectors', 'Token vectors flow through the whole network and each sub-layer adds an update to them', 'Residuals are only used in CNNs', 'Attention removes information'], answer: 1, explain: 'Layers read from and write (add) to a shared stream.' }
    ],
    problems: [
      {
        id: 'ffn-gelu', title: 'The Feed-Forward Network', difficulty: 'easy',
        prompt: `Implement:

1. \`gelu(x)\` — the tanh approximation used by GPT-2:
   0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
2. \`ffn(x, W1, b1, W2, b2)\` = gelu(x @ W1 + b1) @ W2 + b2, applied to every row (token) of x.

\`\`\`text
gelu(np.array([0.0]))  ->  [0.]
\`\`\``,
        starter: `
import numpy as np

def gelu(x):
    # TODO
    pass

def ffn(x, W1, b1, W2, b2):
    # TODO
    pass
`,
        solution: `
import numpy as np

def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn(x, W1, b1, W2, b2):
    return gelu(np.asarray(x, dtype=float) @ W1 + b1) @ W2 + b2
`,
        hints: ['Write gelu exactly as the formula with NumPy functions.', 'The FFN is two dense layers with GELU in between.'],
        explain: `GELU is like a smooth ReLU: it lets small negative values through slightly, which tends to help transformers. Because the FFN uses the same weights for every row, it processes each token's vector independently — no information crosses positions here.`,
        tests: [
          { call: 'gelu(np.array([0.0]))', expect: 'np.array([0.0])' },
          { call: 'gelu(np.array([-3.0, 1.0, 3.0]))', expect: '0.5 * np.array([-3.0, 1.0, 3.0]) * (1 + np.tanh(np.sqrt(2 / np.pi) * (np.array([-3.0, 1.0, 3.0]) + 0.044715 * np.array([-3.0, 1.0, 3.0]) ** 3)))' },
          { name: 'ffn on a random batch', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=(4, 6)); W1 = rng.normal(size=(6, 24)) * 0.3; b1 = rng.normal(size=24) * 0.1; W2 = rng.normal(size=(24, 6)) * 0.3; b2 = rng.normal(size=6) * 0.1', call: 'ffn(x, W1, b1, W2, b2)', hidden: true },
          { name: 'rows are processed independently', setup: 'rng = np.random.default_rng(1)\nx = rng.normal(size=(3, 4)); W1 = rng.normal(size=(4, 8)); b1 = np.zeros(8); W2 = rng.normal(size=(8, 4)); b2 = np.zeros(4)', code: 'full = ffn(x, W1, b1, W2, b2)\nassert np.allclose(full[1], ffn(x[1:2], W1, b1, W2, b2)[0])', hidden: true }
        ]
      },
      {
        id: 'transformer-block-forward', title: 'Assemble a Transformer Block', difficulty: 'hard',
        prompt: `Compose the provided pieces into a **pre-norm** transformer block:

\`\`\`text
h = layer_norm(x, p["ln1_g"], p["ln1_b"])
x = x + mha(h, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, causal=True)
h = layer_norm(x, p["ln2_g"], p["ln2_b"])
x = x + ffn(h, p["W1"], p["b1"], p["W2"], p["b2"])
\`\`\`

Implement \`block(x, p, n_heads)\` (returning the new (T, d) array) and \`stack(x, blocks, n_heads)\` that applies a list of parameter dicts one after another. \`layer_norm\`, \`mha\` and \`ffn\` are provided in the starter.

The hidden tests also check **causality**: changing a later token must not change any earlier output.`,
        starter: `
import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d = X.shape; dh = d // n_heads
    split = lambda M: M.reshape(T, n_heads, dh).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ V).transpose(1, 0, 2).reshape(T, d) @ Wo

def block(x, p, n_heads):
    # TODO: pre-norm attention sub-layer with residual, then pre-norm FFN sub-layer with residual
    pass

def stack(x, blocks, n_heads):
    # TODO
    pass
`,
        solution: `
import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d = X.shape; dh = d // n_heads
    split = lambda M: M.reshape(T, n_heads, dh).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ V).transpose(1, 0, 2).reshape(T, d) @ Wo

def block(x, p, n_heads):
    h = layer_norm(x, p["ln1_g"], p["ln1_b"])
    x = x + mha(h, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, causal=True)
    h = layer_norm(x, p["ln2_g"], p["ln2_b"])
    x = x + ffn(h, p["W1"], p["b1"], p["W2"], p["b2"])
    return x

def stack(x, blocks, n_heads):
    for p in blocks:
        x = block(x, p, n_heads)
    return x
`,
        hints: ['Normalize first, then add the sub-layer output to the *un-normalized* x (the residual).', 'Use the second LayerNorm on the updated x before the FFN.'],
        explain: `The residual additions mean each sub-layer only writes a delta into the stream. Pre-norm (LayerNorm inside the residual branch) keeps the stream itself un-normalized, which is what makes deep stacks train stably.`,
        tests: [
          { name: 'one block vs reference', setup: 'rng = np.random.default_rng(0)\nd = 8\ndef params(seed):\n    r = np.random.default_rng(seed)\n    return {"ln1_g": np.ones(d), "ln1_b": np.zeros(d), "ln2_g": np.ones(d), "ln2_b": np.zeros(d), "Wq": r.normal(0, 0.3, (d, d)), "Wk": r.normal(0, 0.3, (d, d)), "Wv": r.normal(0, 0.3, (d, d)), "Wo": r.normal(0, 0.3, (d, d)), "W1": r.normal(0, 0.3, (d, 4 * d)), "b1": np.zeros(4 * d), "W2": r.normal(0, 0.3, (4 * d, d)), "b2": np.zeros(d)}\nx = rng.normal(size=(5, d))', call: 'block(x, params(1), 2)' },
          { name: 'two-block stack', setup: 'rng = np.random.default_rng(3)\nd = 8\ndef params(seed):\n    r = np.random.default_rng(seed)\n    return {"ln1_g": r.normal(1, 0.1, d), "ln1_b": r.normal(0, 0.1, d), "ln2_g": r.normal(1, 0.1, d), "ln2_b": r.normal(0, 0.1, d), "Wq": r.normal(0, 0.3, (d, d)), "Wk": r.normal(0, 0.3, (d, d)), "Wv": r.normal(0, 0.3, (d, d)), "Wo": r.normal(0, 0.3, (d, d)), "W1": r.normal(0, 0.3, (d, 4 * d)), "b1": r.normal(0, 0.1, 4 * d), "W2": r.normal(0, 0.3, (4 * d, d)), "b2": r.normal(0, 0.1, d)}\nx = rng.normal(size=(6, d))', call: 'stack(x, [params(10), params(11)], 4)', hidden: true },
          { name: 'causal and shape-preserving', setup: 'rng = np.random.default_rng(5)\nd = 8\nr = np.random.default_rng(7)\np = {"ln1_g": np.ones(d), "ln1_b": np.zeros(d), "ln2_g": np.ones(d), "ln2_b": np.zeros(d), "Wq": r.normal(0, 0.3, (d, d)), "Wk": r.normal(0, 0.3, (d, d)), "Wv": r.normal(0, 0.3, (d, d)), "Wo": r.normal(0, 0.3, (d, d)), "W1": r.normal(0, 0.3, (d, 4 * d)), "b1": np.zeros(4 * d), "W2": r.normal(0, 0.3, (4 * d, d)), "b2": np.zeros(d)}\nx = rng.normal(size=(6, d))', code: 'a = stack(x, [p, p], 2)\nassert a.shape == x.shape\nx2 = x.copy(); x2[5] = rng.normal(size=d)\nb = stack(x2, [p, p], 2)\nassert np.allclose(a[:5], b[:5]), "earlier positions must not see later tokens"', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Jay Alammar — The Illustrated GPT-2', url: 'https://jalammar.github.io/illustrated-gpt2/' },
      { title: 'Anthropic — A Mathematical Framework for Transformer Circuits (residual stream)', url: 'https://transformer-circuits.pub/2021/framework/index.html' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
