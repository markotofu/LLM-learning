/* ML Atlas – content: Python & Data */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'python-essentials', branch: 'code', title: 'Python Essentials for Machine Learning', short: 'Python Essentials', level: 1, minutes: 45, prereqs: ['what-is-ml'],
    tags: ['python', 'lists', 'dicts', 'functions', 'classes', 'comprehensions'],
    summary: 'Just enough Python to read and write ML code: values, lists and dicts, loops, functions, comprehensions and classes.',
    why: `Python is the language of machine learning: PyTorch, TensorFlow, JAX, scikit-learn, pandas and Hugging Face are all Python-first. You do not need to master the whole language — ML code leans heavily on a small core: lists and dictionaries, loops, functions, comprehensions and classes (every PyTorch model is a class). This lesson covers exactly that core, with the ML-flavoured examples you will meet again later, like building a vocabulary for a tokenizer.`,
    objectives: [
      'Use numbers, strings, booleans, None and f-strings',
      'Work with lists, tuples, dictionaries and sets, including slicing',
      'Write loops, conditions and functions, and use enumerate / zip / sorted',
      'Read and write list and dict comprehensions',
      'Define a small class with __init__ and methods, as ML libraries do'
    ],
    sections: [
      { type: 'text', title: 'Values and variables', md: `A **variable** is a name for a value. Python figures out the type for you:

| Type | Examples | Notes |
| --- | --- | --- |
| \`int\` | \`3\`, \`-42\`, \`10_000\` | whole numbers of any size |
| \`float\` | \`3.14\`, \`1e-5\` | decimals; \`1e-5\` = 0.00001 (learning rates!) |
| \`str\` | \`"cat"\`, \`'hello world'\` | text |
| \`bool\` | \`True\`, \`False\` | results of comparisons |
| \`None\` | \`None\` | "no value" — what a function without \`return\` gives back |

Arithmetic you'll use constantly: \`+ - * /\`, \`**\` (power), \`//\` (integer division), \`%\` (remainder). Note that \`/\` always gives a float: \`7 / 2 == 3.5\`.

**f-strings** put values into text: \`f"loss = {loss:.4f}"\` prints the loss with 4 decimals — you will see this in every training loop.` },
      { type: 'code', title: 'Basics in action', code: `
learning_rate = 1e-3          # scientific notation: 0.001
epochs = 5
loss = 0.693147
model_name = "tiny-gpt"
is_training = True

print(type(learning_rate), type(epochs), type(model_name), type(is_training))
print(7 / 2, 7 // 2, 7 % 2, 2 ** 10)          # 3.5, 3, 1, 1024
print(f"{model_name}: epoch {epochs}, loss = {loss:.3f}, lr = {learning_rate:g}")

def no_return():
    x = 1                     # forgot to return!
print("a function without return gives:", no_return())
` },
      { type: 'text', title: 'Collections: lists, tuples, dicts and sets', md: `**Lists** are ordered and changeable: \`scores = [0.9, 0.7, 0.8]\`.
- Indexing starts at **0**: \`scores[0]\` is the first item, \`scores[-1]\` the last.
- **Slicing** \`scores[start:stop]\` takes items from \`start\` up to (not including) \`stop\`: \`scores[0:2] == [0.9, 0.7]\`. \`scores[::-1]\` reverses.
- \`scores.append(0.95)\` adds to the end; \`len(scores)\` counts.

**Tuples** are like lists but cannot change: \`shape = (32, 784)\`. Array shapes are tuples.

**Dictionaries** map keys to values: \`config = {"lr": 0.001, "epochs": 10}\`, \`config["lr"]\`. They are everywhere in ML: hyperparameter configs, a tokenizer's vocabulary (\`{"the": 0, "cat": 1}\`), counts of words.

**Sets** hold unique items: \`set(["a", "b", "a"]) == {"a", "b"}\` — handy for finding the distinct words in a text.` },
      { type: 'code', title: 'Building a vocabulary (the first step of a tokenizer)', code: `
text = "the cat sat on the mat the end"
words = text.split()                       # split on whitespace -> list of strings
print(words)
print("first:", words[0], " last:", words[-1], " slice:", words[1:4])

vocab = sorted(set(words))                 # unique words, alphabetical
print("vocabulary:", vocab)

word_to_id = {w: i for i, w in enumerate(vocab)}   # dict comprehension
print("word_to_id:", word_to_id)

ids = [word_to_id[w] for w in words]       # encode the text as integers
print("encoded:", ids)

id_to_word = {i: w for w, i in word_to_id.items()}
print("decoded:", " ".join(id_to_word[i] for i in ids))
`, after: `This is a miniature version of what every LLM tokenizer does: map pieces of text to integer ids (encode) and back (decode). Models only ever see the integers.` },
      { type: 'check', question: { type: 'fill', q: 'What does `[10, 20, 30, 40, 50][1:3]` evaluate to?', answer: ['[20, 30]', '[20,30]'], display: '[20, 30]', explain: 'Index 1 is 20; the slice stops *before* index 3, so you get items at 1 and 2.' } },
      { type: 'text', title: 'Control flow and functions', md: `Python groups code by **indentation** (4 spaces is the convention). Everything indented under a \`def\`, \`if\` or \`for\` belongs to it.

\`\`\`python
def accuracy(predictions, labels):
    """Fraction of predictions that match the labels."""
    correct = 0
    for p, y in zip(predictions, labels):   # walk two lists in step
        if p == y:
            correct += 1
    return correct / len(labels)
\`\`\`

Loop helpers you will use constantly:
- \`range(n)\` — 0, 1, …, n−1 (\`for epoch in range(10):\`).
- \`enumerate(items)\` — gives \`(index, item)\` pairs.
- \`zip(a, b)\` — pairs up items from two lists.
- \`sorted(items, key=..., reverse=True)\` — sort by any criterion.

Functions can have **default arguments** — \`def train(lr=0.01, epochs=10):\` — and are called with keywords for clarity: \`train(epochs=5)\`.` },
      { type: 'code', title: 'Comprehensions, sorting and lambdas', md: `A **list comprehension** builds a list in one readable line: \`[expression for item in items if condition]\`.`, code: `
scores = {"alice": 0.91, "bob": 0.78, "carol": 0.88, "dan": 0.65}

passed = [name for name, s in scores.items() if s >= 0.8]
print("passed:", passed)

squared = [x ** 2 for x in range(6)]
print("squares:", squared)

ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)   # sort by score
print("ranking:", ranked)

for rank, (name, s) in enumerate(ranked, start=1):
    print(f"{rank}. {name:<6} {s:.0%}")
` },
      { type: 'text', title: 'Classes: bundling data with behaviour', md: `A **class** groups data (attributes) with functions that use it (methods). ML libraries use classes everywhere: a model *holds* its parameters and *knows how* to make predictions.

\`\`\`python
class LinearModel:
    def __init__(self, w, b):      # runs when you create a model
        self.w = w                 # self = "this particular object"
        self.b = b

    def predict(self, x):
        return self.w * x + self.b

model = LinearModel(w=2.0, b=1.0)
model.predict(3.0)                 # 7.0
\`\`\`

\`self\` is how a method reaches the object's own data. When you later write a PyTorch model, you'll subclass \`nn.Module\`, create layers in \`__init__\` and define the computation in a method called \`forward\` — the same pattern.` },
      { type: 'code', title: 'A tiny model class that learns', code: `
class MeanPredictor:
    """The simplest possible regression model: always predict the training mean.
    It is the baseline every real model must beat."""

    def fit(self, ys):
        self.mean_ = sum(ys) / len(ys)      # trailing underscore = learned from data (sklearn style)
        return self                         # returning self allows chaining

    def predict(self, n):
        return [self.mean_] * n

prices = [210, 340, 190, 280, 305]
model = MeanPredictor().fit(prices)
print("learned mean:", model.mean_)
print("predictions:", model.predict(3))
` },
      { type: 'check', question: { type: 'mcq', q: 'Inside a class method, what is `self`?', options: ['A keyword that makes the method run faster', 'The particular object the method was called on', 'The class\'s parent class', 'A global variable'], answer: 1, explain: '`model.predict(3)` calls `predict` with `self` set to `model`, so the method can read `self.mean_` for that specific object.' } },
      { type: 'text', title: 'Python habits for ML code', md: `- **Let libraries do the number crunching.** A Python \`for\` loop over a million numbers is ~100× slower than the equivalent NumPy operation. Use plain Python for logic and orchestration, NumPy/PyTorch for math.
- **Don't modify inputs by surprise.** If a function needs to change a list or array, work on a copy (\`xs = list(xs)\` or \`arr.copy()\`).
- **Read tracebacks from the bottom up.** The last line says what went wrong (\`KeyError: 'lr'\`); the lines above show where.
- **Print shapes and types** when confused: \`print(type(x), len(x))\`.
- **Seed your randomness** so experiments are reproducible.` }
    ],
    when: `- Use **lists** for ordered collections you build up step by step; **dicts** for lookups by name or id (configs, vocabularies, counts); **sets** for uniqueness and fast membership tests.
- Use **comprehensions** for short transformations; switch to a regular loop when the logic needs several lines.
- Use **classes** when data and the operations on it belong together (models, datasets, tokenizers); plain functions otherwise.
- Use **NumPy** (next lesson) as soon as you are doing arithmetic on many numbers.`,
    pitfalls: `- **Off-by-one slicing** — \`xs[a:b]\` excludes index b.
- **Forgetting \`return\`** — the function silently returns \`None\`.
- **Mutable default arguments** — \`def f(xs=[])\` shares one list across calls; use \`xs=None\` and create the list inside.
- **Integer vs float division** — \`//\` truncates.
- **Mixing tabs and spaces** — indentation errors. Use 4 spaces.
- **\`[[0] * 3] * 3\`** — three references to the *same* row; use \`[[0] * 3 for _ in range(3)]\`.`,
    takeaways: [
      'Indexing starts at 0; slices xs[a:b] stop before b.',
      'Dicts map keys to values — configs, vocabularies and counts are all dicts.',
      'zip, enumerate, sorted(key=...) and comprehensions make data-wrangling code short and clear.',
      'A class bundles data (self.attributes) with methods; ML models follow this pattern.',
      'Use Python for logic, NumPy/PyTorch for heavy arithmetic.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does `len({"a": 1, "b": 2, "c": 3})` return?', options: ['6', '3', '2', 'An error'], answer: 1, explain: 'len of a dict counts its keys: 3.' },
      { type: 'fill', q: 'What does `[x * 2 for x in range(4)]` produce?', answer: ['[0, 2, 4, 6]', '[0,2,4,6]'], display: '[0, 2, 4, 6]', explain: 'range(4) is 0, 1, 2, 3; each doubled.' },
      { type: 'mcq', q: 'A function has no `return` statement. What does calling it return?', options: ['0', 'An empty string', 'None', 'An error'], answer: 2, explain: 'Python functions return None by default — a very common source of bugs in practice problems.' },
      { type: 'mcq', q: 'Which data structure is best for looking up a word\'s integer id in a tokenizer?', options: ['A list of words', 'A dict mapping word → id', 'A tuple', 'A set'], answer: 1, explain: 'Dicts give fast lookup by key: word_to_id["cat"].' },
      { type: 'tf', q: '`xs[-1]` gives the last element of a non-empty list xs.', answer: true, explain: 'Negative indices count from the end: -1 is the last item, -2 the second-to-last.' },
      { type: 'mcq', q: 'Why do ML libraries wrap models in classes?', options: ['Classes run faster than functions', 'A model needs to keep its parameters (data) together with its methods like predict/forward', 'Python requires it', 'To avoid using dictionaries'], answer: 1, explain: 'The parameters live on the object (self.w) and the methods use them — the PyTorch nn.Module pattern.' }
    ],
    problems: [
      {
        id: 'word-frequencies', title: 'Word Frequencies', difficulty: 'easy',
        prompt: `Counting words is the first step of many NLP pipelines (bag-of-words, vocabularies, n-gram models). Implement two functions:

1. \`word_counts(text)\` → a dict mapping each word to how many times it appears.
   - Lowercase everything.
   - Split on whitespace (\`text.split()\`).
   - Strip these punctuation characters from both ends of each word: \`.,!?;:"'()\` (use \`word.strip(...)\`).
   - Ignore anything that becomes empty.
2. \`top_k(counts, k)\` → a list of the k most frequent \`(word, count)\` tuples, sorted by count (highest first) and, for ties, alphabetically.

\`\`\`text
word_counts("The cat. The hat!")  ->  {"the": 2, "cat": 1, "hat": 1}
top_k({"the": 2, "cat": 1, "hat": 1}, 2)  ->  [("the", 2), ("cat", 1)]
\`\`\``,
        starter: `
PUNCT = ".,!?;:\\"'()"

def word_counts(text):
    """Map each lowercase, punctuation-stripped word to its count."""
    counts = {}
    # TODO
    return counts


def top_k(counts, k):
    """The k most common (word, count) pairs: count descending, then word ascending."""
    # TODO: sorted(..., key=...) is your friend
    pass
`,
        solution: `
PUNCT = ".,!?;:\\"'()"

def word_counts(text):
    counts = {}
    for raw in text.lower().split():
        word = raw.strip(PUNCT)
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


def top_k(counts, k):
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
`,
        hints: ['`counts.get(word, 0) + 1` increments a count that may not exist yet.', 'Sort by a tuple key: `key=lambda kv: (-kv[1], kv[0])` — negative count gives descending counts, the word breaks ties alphabetically.', 'Slice the sorted list with `[:k]`.'],
        explain: `\`dict.get(key, default)\` avoids a KeyError the first time a word appears. Sorting by the tuple \`(-count, word)\` handles "highest first, then alphabetical" in one pass. Python's \`collections.Counter\` does this for you (\`Counter(words).most_common(k)\`), but its tie order is insertion order — writing the key yourself makes the result deterministic.`,
        tests: [
          { call: 'word_counts("The cat. The hat!")', expect: '{"the": 2, "cat": 1, "hat": 1}' },
          { call: 'top_k({"the": 2, "cat": 1, "hat": 1}, 2)', expect: '[("the", 2), ("cat", 1)]' },
          { name: 'punctuation and quotes', call: 'word_counts("\\"Hello,\\" she said. Hello again (hello)!")', expect: '{"hello": 3, "she": 1, "said": 1, "again": 1}' },
          { name: 'ties sorted alphabetically', call: 'top_k({"b": 2, "a": 2, "c": 5, "d": 1}, 3)', expect: '[("c", 5), ("a", 2), ("b", 2)]', hidden: true },
          { name: 'empty and punctuation-only words', call: 'word_counts("  ... !! ok  ")', expect: '{"ok": 1}', hidden: true },
          { name: 'longer text', setup: 'text = "To be, or not to be: that is the question. Whether tis nobler in the mind to suffer"', call: 'top_k(word_counts(text), 4)', hidden: true }
        ]
      },
      {
        id: 'moving-average', title: 'Moving Average', difficulty: 'easy',
        prompt: `A moving average smooths a noisy series — the same idea is used to smooth training-loss curves and in the "momentum" of optimizers.

Implement \`moving_average(xs, window)\` returning a list with the average of every run of \`window\` consecutive values. The result has \`len(xs) - window + 1\` items.

- If \`window < 1\` or \`window > len(xs)\`, \`raise ValueError\`.
- Use plain Python (no NumPy).

\`\`\`text
moving_average([1, 2, 3, 4, 5], 2)  ->  [1.5, 2.5, 3.5, 4.5]
moving_average([1, 2, 3, 4, 5], 5)  ->  [3.0]
\`\`\``,
        starter: `
def moving_average(xs, window):
    """Averages of each consecutive run of 'window' values."""
    # TODO: validate window, then loop over the start positions
    pass
`,
        solution: `
def moving_average(xs, window):
    if window < 1 or window > len(xs):
        raise ValueError("window must be between 1 and len(xs)")
    out = []
    running = sum(xs[:window])
    out.append(running / window)
    for i in range(window, len(xs)):
        running += xs[i] - xs[i - window]   # slide: add the new value, drop the oldest
        out.append(running / window)
    return out
`,
        hints: ['The windows start at positions 0, 1, …, len(xs) − window.', 'A simple version: `[sum(xs[i:i + window]) / window for i in range(len(xs) - window + 1)]`.', 'Faster: keep a running sum — add the value entering the window and subtract the one leaving.'],
        explain: `The comprehension version is correct but re-adds every window from scratch (O(n·w)). The sliding running sum updates in O(1) per step: add the newcomer, subtract the value that fell out. "Maintain a running statistic instead of recomputing" is a pattern you will see again in exponential moving averages and optimizers.`,
        tests: [
          { call: 'moving_average([1, 2, 3, 4, 5], 2)', expect: '[1.5, 2.5, 3.5, 4.5]' },
          { call: 'moving_average([1, 2, 3, 4, 5], 5)', expect: '[3.0]' },
          { call: 'moving_average([4, 8, 6, -1, 3], 3)', expect: '[6.0, 13 / 3, 8 / 3]' },
          { name: 'window of 1 returns the values', call: 'moving_average([3, 1, 2], 1)', expect: '[3.0, 1.0, 2.0]', hidden: true },
          { name: 'invalid window raises ValueError', code: 'for bad in (0, 6):\n    try:\n        moving_average([1, 2, 3, 4, 5], bad)\n    except ValueError:\n        continue\n    raise AssertionError("window=%d should raise ValueError" % bad)', hidden: true },
          { name: 'long random series', setup: 'rng = np.random.default_rng(1)\nxs = rng.normal(size=200).tolist()', call: 'moving_average(xs, 7)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'The official Python tutorial', url: 'https://docs.python.org/3/tutorial/' },
      { title: "Python documentation — Built-in types", url: "https://docs.python.org/3/library/stdtypes.html" }
    ]
  },

  // =====================================================================================
  {
    id: 'numpy', branch: 'code', title: 'NumPy: Arrays, Broadcasting & Vectorization', short: 'NumPy', level: 1, minutes: 45, prereqs: ['python-essentials', 'vectors-matrices'],
    tags: ['numpy', 'arrays', 'broadcasting', 'vectorization', 'axis'],
    summary: 'The array library underneath all of Python ML: shapes, indexing, axis-wise operations, broadcasting, and why vectorized code is ~100× faster.',
    why: `NumPy's \`ndarray\` is the data structure of scientific Python. pandas columns, scikit-learn inputs and images are NumPy arrays, and PyTorch tensors deliberately copy NumPy's API — learn NumPy once and you can read all of them. More importantly, NumPy teaches you to think in **whole arrays**: one expression that operates on a million numbers at C speed instead of a Python loop. That habit — vectorization — is how all efficient ML code (and GPU code) is written.`,
    objectives: [
      'Create arrays and reason about their shape, dtype and number of dimensions',
      'Index and slice arrays, filter with boolean masks, and know when you get a view vs a copy',
      'Aggregate along an axis (sum, mean, argmax) and predict the resulting shape',
      'Apply broadcasting rules to combine arrays of different shapes',
      'Replace Python loops with vectorized expressions'
    ],
    sections: [
      { type: 'text', title: 'Why NumPy?', md: `A Python list of numbers is a list of pointers to separate number objects scattered in memory; every \`+\` goes through Python's machinery. A NumPy array stores raw numbers of **one type** (\`float64\`, \`int32\`, …) **contiguously**, and its operations run in compiled C (often using your CPU's vector instructions).

The result: \`a + b\` on a million numbers takes a millisecond or two in NumPy versus roughly a hundred milliseconds with a Python loop.

Three attributes describe every array:
- \`.shape\` — size along each dimension, e.g. \`(32, 784)\`;
- \`.ndim\` — number of dimensions (2 for a matrix);
- \`.dtype\` — the element type (\`float64\`, \`int64\`, \`bool\`…).` },
      { type: 'code', title: 'Creating and inspecting arrays', code: `
import numpy as np

a = np.array([1.0, 2.0, 3.0])
print(a, a.shape, a.ndim, a.dtype)

X = np.arange(12).reshape(3, 4)        # 0..11 arranged as 3 rows x 4 columns
print(X)
print("shape:", X.shape)

print(np.zeros((2, 3)))                 # common initialisers
print(np.ones(4))
print(np.linspace(0, 1, 5))             # 5 evenly spaced points from 0 to 1

rng = np.random.default_rng(0)          # the modern way to get random numbers
W = rng.normal(0, 0.1, size=(3, 2))     # e.g. initial weights of a small layer
print(W.round(3), W.dtype)
print(X.astype(np.float32).dtype)       # change the type (GPUs love float32)
` },
      { type: 'text', title: 'Indexing, slicing and masks', md: `For a 2-D array \`X\`:

| Expression | Meaning | Result shape for X (5, 3) |
| --- | --- | --- |
| \`X[2, 1]\` | one element (row 2, column 1) | scalar |
| \`X[2]\` or \`X[2, :]\` | row 2 | (3,) |
| \`X[:, 0]\` | column 0 (the first feature of every example) | (5,) |
| \`X[1:3]\` | rows 1 and 2 | (2, 3) |
| \`X[X > 0]\` | all positive elements (boolean mask) | (k,) |
| \`X[[0, 4]]\` | rows 0 and 4 (fancy indexing) | (2, 3) |

**Boolean masks** are how you filter data: \`y == 1\` gives an array of True/False, and \`X[y == 1]\` selects the rows of the positive class.

:::warn Slices are views, not copies
\`row = X[0]\` does not copy anything — it is a window into X. Changing \`row[0] = 99\` changes \`X\` too! Use \`X[0].copy()\` when you need an independent array. (Masks and fancy indexing *do* return copies.)
:::` },
      { type: 'code', title: 'Indexing in practice', code: `
import numpy as np

X = np.array([[ 5.1, 3.5, 0],
              [ 4.9, 3.0, 1],
              [ 6.2, 2.9, 1],
              [ 5.9, 3.0, 0]])
features, labels = X[:, :2], X[:, 2]
print("features shape:", features.shape, " labels:", labels)

print("rows with label 1:")
print(features[labels == 1])            # boolean mask selects rows
print("first column > 5:", features[:, 0] > 5)

view = X[0]                             # a view into X
view[0] = 99.0
print("X[0, 0] after changing the view:", X[0, 0])   # it changed X too!
safe = X[1].copy()
safe[0] = -1.0
print("X[1, 0] after changing a copy:", X[1, 0])     # unchanged
` },
      { type: 'check', question: { type: 'fill', q: 'X has shape (5, 3). What is the shape of `X[:, 1]`? (write it like (n,))', answer: ['(5,)', '(5, )', '5'], display: '(5,)', explain: 'Selecting one column with a plain integer index removes that dimension: 5 rows → a 1-D array of length 5.' } },
      { type: 'text', title: 'Aggregations and the axis argument', md: `\`sum\`, \`mean\`, \`std\`, \`max\`, \`argmax\` and friends work on the whole array, or along one **axis**:

- \`axis=0\` — collapse the rows → one result **per column** (e.g. the mean of every feature).
- \`axis=1\` — collapse the columns → one result **per row** (e.g. the predicted class of every example with \`argmax\`).

Memory trick: **the axis you name is the one that disappears.** For X with shape (100, 10), \`X.mean(axis=0)\` has shape (10,) and \`X.mean(axis=1)\` has shape (100,).

Add \`keepdims=True\` to keep the collapsed axis with size 1 — shape (100, 1) — which is exactly what you need for the next topic, broadcasting.` },
      { type: 'text', title: 'Broadcasting', md: `Broadcasting lets NumPy combine arrays of different shapes **without copying data**. Compare the shapes from the right-hand end; two dimensions are compatible if they are **equal** or one of them is **1** (missing dimensions count as 1). The size-1 dimension is stretched to match.

| A | B | Result | Use |
| --- | --- | --- | --- |
| (100, 10) | (10,) | (100, 10) | subtract each column's mean |
| (100, 10) | (100, 1) | (100, 10) | divide each row by its own norm |
| (3, 1) | (4,) | (3, 4) | an "outer" combination of every pair |
| (n, 1, d) | (1, m, d) | (n, m, d) | differences between every pair of points |
| (100, 10) | (100,) | ✗ error | 10 vs 100 don't match — reshape to (100, 1) |

Broadcasting is how you standardize a dataset in one line, \`(X - X.mean(axis=0)) / X.std(axis=0)\`, and how you compute all pairwise distances without a loop.` },
      { type: 'code', title: 'Broadcasting and the speed of vectorization', code: `
import numpy as np, time

rng = np.random.default_rng(1)
X = rng.normal(5, 2, size=(1000, 4))

Z = (X - X.mean(axis=0)) / X.std(axis=0)      # (1000,4) op (4,) -> standardize columns
print("column means after:", Z.mean(axis=0).round(6))
print("column stds after: ", Z.std(axis=0).round(6))

norms = np.linalg.norm(X, axis=1, keepdims=True)   # (1000, 1)
unit = X / norms                                   # every row now has length 1
print("row lengths:", np.linalg.norm(unit, axis=1)[:3])

# Speed: Python loop vs vectorized NumPy for the same computation
a = rng.random(200_000); b = rng.random(200_000)
t0 = time.perf_counter(); s = 0.0
for i in range(len(a)):
    s += a[i] * b[i]
t1 = time.perf_counter(); s2 = a @ b; t2 = time.perf_counter()
print(f"python loop: {t1 - t0:.4f}s   numpy: {t2 - t1:.6f}s   same result: {np.isclose(s, s2)}")
` },
      { type: 'check', question: { type: 'fill', q: 'What is the shape of `np.ones((3, 1)) + np.ones(4)`? (write it like (a, b))', answer: ['(3, 4)', '(3,4)'], display: '(3, 4)', explain: 'Align from the right: (3, 1) vs (·, 4). 1 stretches to 4 and the missing dimension stretches to 3 → (3, 4).' } }
    ],
    when: `- Use NumPy for **any** numeric work on more than a handful of numbers.
- Reach for **axis** arguments instead of looping over rows or columns.
- Use **broadcasting** to apply per-feature or per-example statistics.
- Keep plain Python for control flow (which file to load, which model to try); move to pandas for labelled tabular data with mixed types, and to PyTorch/JAX when you need GPUs or automatic gradients.`,
    pitfalls: `- **Views vs copies** — modifying a slice modifies the original.
- **Shape (n,) vs (n, 1)** — a 1-D array and a column vector broadcast very differently; \`keepdims=True\` or \`[:, None]\` fixes many surprises.
- **Integer arrays** — \`np.array([1, 2]) / 2\` is fine, but in-place \`a /= 2\` on an int array fails; create float arrays when doing math.
- **Silent broadcasting** — a shape mistake can broadcast into a huge wrong array instead of raising an error. Assert shapes in important code.
- **The old \`np.random.seed\` global state** — prefer \`rng = np.random.default_rng(seed)\`.`,
    takeaways: [
      'An ndarray has a shape, a dtype and stores numbers contiguously — operations run in C.',
      'X[:, j] is a column, X[i] is a row, X[mask] filters rows; slices are views.',
      'Aggregations take an axis: the named axis disappears from the result.',
      'Broadcasting aligns shapes from the right; size-1 dimensions stretch.',
      'Vectorized code is shorter and roughly 100× faster than Python loops.'
    ],
    quiz: [
      { type: 'mcq', q: 'X has shape (64, 10). What is the shape of `X.mean(axis=0)`?', options: ['(64,)', '(10,)', '(64, 10)', '()'], answer: 1, explain: 'axis=0 collapses the 64 rows, leaving one mean per column: (10,).' },
      { type: 'mcq', q: 'Which shapes broadcast together?', options: ['(5, 3) and (5,)', '(5, 3) and (3,)', '(5, 3) and (2, 3)', '(4,) and (5,)'], answer: 1, explain: 'Aligned from the right, (3,) matches the last dimension of (5, 3). (5,) would be compared with 3 and fail.' },
      { type: 'tf', q: 'After `r = X[0]; r[0] = 7`, the array X is unchanged.', answer: false, explain: 'Basic slicing returns a view, so writing into r writes into X. Use X[0].copy() for an independent array.' },
      { type: 'fill', q: 'Logits have shape (32, 10). Which axis do you pass to `argmax` to get the predicted class of each of the 32 examples?', answer: ['1', 'axis=1', '-1', 'axis=-1'], display: '1 (or -1)', explain: 'Collapse the class dimension (axis 1), leaving one index per example: shape (32,).' },
      { type: 'mcq', q: 'Why is `a @ b` so much faster than a Python for-loop computing the same dot product?', options: ['It uses a different algorithm with fewer multiplications', 'It runs in compiled code over contiguous typed memory, avoiding Python\'s per-element overhead', 'It skips small numbers', 'It caches the result'], answer: 1, explain: 'Same arithmetic, but no interpreter overhead per element, plus CPU vector instructions.' }
    ],
    problems: [
      {
        id: 'normalize-rows', title: 'Normalize Rows to Unit Length', difficulty: 'easy',
        prompt: `Implement \`normalize_rows(X)\` for a 2-D NumPy array: divide every row by its L2 norm (length) so that each row has length 1.

- Rows that are all zeros (norm 0) must stay all zeros — no division by zero, no NaN.
- Do not modify \`X\` in place; return a new float array of the same shape.
- Try to do it **without a Python loop**: \`np.linalg.norm(X, axis=1, keepdims=True)\` + broadcasting.

\`\`\`text
normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))  ->  [[0.6, 0.8], [0.0, 1.0]]
\`\`\`

(Unit-length embeddings make cosine similarity a plain dot product — that's how vector databases are usually set up.)`,
        starter: `
import numpy as np

def normalize_rows(X):
    """Scale each row of X to unit L2 norm; all-zero rows stay zero."""
    X = np.asarray(X, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def normalize_rows(X):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)   # shape (n, 1)
    safe = np.where(norms == 0, 1.0, norms)              # avoid dividing by zero
    return X / safe
`,
        hints: ['`np.linalg.norm(X, axis=1, keepdims=True)` has shape (n, 1), which broadcasts across each row.', 'Replace zero norms with 1 before dividing: `np.where(norms == 0, 1.0, norms)`. A zero row divided by 1 is still zero.'],
        explain: `\`keepdims=True\` keeps the norms as a column of shape (n, 1), so \`X / norms\` divides every element of row i by row i's norm. \`np.where\` swaps zero norms for 1, which leaves zero rows unchanged and avoids NaNs (0/0). No loops needed.`,
        tests: [
          { call: 'normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))', expect: 'np.array([[0.6, 0.8], [0.0, 1.0]])' },
          { name: 'zero row stays zero', call: 'normalize_rows(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]))', expect: 'np.array([[0.0, 0.0, 0.0], [1/3, 2/3, 2/3]])' },
          { name: 'input not modified', code: 'X = np.array([[3.0, 4.0]])\nnormalize_rows(X)\nassert np.allclose(X, [[3.0, 4.0]]), "normalize_rows must not change its input"' },
          { name: 'random matrix: every row has length 1', setup: 'rng = np.random.default_rng(3)\nX = rng.normal(size=(50, 8))', code: 'out = normalize_rows(X)\nassert out.shape == X.shape\nassert np.allclose(np.linalg.norm(out, axis=1), 1.0)\nassert np.allclose(out * np.linalg.norm(X, axis=1, keepdims=True), X)', hidden: true },
          { name: 'integer input', call: 'normalize_rows(np.array([[6, 8], [5, 0]]))', expect: 'np.array([[0.6, 0.8], [1.0, 0.0]])', hidden: true }
        ]
      },
      {
        id: 'pairwise-distances', title: 'Pairwise Distances with Broadcasting', difficulty: 'medium',
        prompt: `Implement \`pairwise_distances(A, B)\` returning the matrix of **Euclidean distances** between every row of \`A\` (shape n×d) and every row of \`B\` (shape m×d). The result has shape **(n, m)** and \`D[i, j] = ‖A[i] − B[j]‖\`.

Use broadcasting, not loops: \`A[:, None, :] - B[None, :, :]\` has shape (n, m, d) — the difference between every pair.

\`\`\`text
pairwise_distances(np.array([[0, 0], [3, 4]]), np.array([[0, 0], [6, 8]]))
-> [[ 0., 10.],
    [ 5.,  5.]]
\`\`\`

This is the core computation of k-nearest neighbours and k-means, which you will build soon.`,
        starter: `
import numpy as np

def pairwise_distances(A, B):
    """(n, m) matrix of Euclidean distances between rows of A (n, d) and B (m, d)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # TODO: broadcast to (n, m, d), square, sum over the last axis, square root
    pass
`,
        solution: `
import numpy as np

def pairwise_distances(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]        # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (n, m)
`,
        hints: ['`A[:, None, :]` inserts a new axis, giving shape (n, 1, d). `B[None, :, :]` gives (1, m, d).', 'Subtracting broadcasts to (n, m, d). Square, `sum(axis=-1)` over d, then `np.sqrt`.'],
        explain: `Inserting size-1 axes with \`None\` lines the arrays up so that broadcasting forms every (i, j) pair at once. For large n and m the (n, m, d) intermediate can use a lot of memory; libraries use the identity ‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b to compute the same matrix from one matrix product, \`A @ B.T\`.`,
        tests: [
          { call: 'pairwise_distances(np.array([[0, 0], [3, 4]]), np.array([[0, 0], [6, 8]]))', expect: 'np.array([[0.0, 10.0], [5.0, 5.0]])' },
          { name: 'shape (n, m)', code: 'D = pairwise_distances(np.zeros((4, 3)), np.ones((7, 3)))\nassert D.shape == (4, 7), "expected shape (4, 7), got %s" % (D.shape,)\nassert np.allclose(D, np.sqrt(3))' },
          { name: '1-D points', call: 'pairwise_distances(np.array([[1.0], [4.0]]), np.array([[2.0], [0.0], [10.0]]))', expect: 'np.array([[1.0, 1.0, 9.0], [2.0, 4.0, 6.0]])' },
          { name: 'random points vs explicit loops', setup: 'rng = np.random.default_rng(8)\nA = rng.normal(size=(6, 5))\nB = rng.normal(size=(9, 5))', call: 'pairwise_distances(A, B)', expect: 'np.array([[np.linalg.norm(a - b) for b in B] for a in A])', hidden: true },
          { name: 'distance of a set to itself has a zero diagonal', setup: 'rng = np.random.default_rng(2)\nA = rng.normal(size=(5, 3))', code: 'D = pairwise_distances(A, A)\nassert np.allclose(np.diag(D), 0)\nassert np.allclose(D, D.T)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'NumPy: the absolute basics for beginners', url: 'https://numpy.org/doc/stable/user/absolute_beginners.html' },
      { title: 'NumPy broadcasting rules', url: 'https://numpy.org/doc/stable/user/basics.broadcasting.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'data-preparation', branch: 'code', title: 'Data Preparation: Cleaning, Splitting & Leakage', short: 'Data Preparation', level: 1, minutes: 40, prereqs: ['numpy', 'statistics', 'pandas-eda'],
    tags: ['pandas', 'missing values', 'train test split', 'data leakage', 'cleaning'],
    summary: 'Turning raw data into trustworthy training data: inspecting it, handling missing values and duplicates, splitting into train/validation/test, and avoiding data leakage.',
    why: `Practitioners routinely say that most of machine learning is data work — and that the quality of the data matters more than the choice of model. A brilliant model trained on messy data learns the mess. Worse, a subtle mistake in how you **split** or **preprocess** data — *data leakage* — can make a useless model look excellent in your notebook and then fail in production. This lesson is about getting the boring parts exactly right, because everything downstream depends on them.`,
    objectives: [
      'Inspect a dataset for types, missing values, duplicates and suspicious values',
      'Choose a strategy for missing values and apply it without leaking information',
      'Split data into training, validation and test sets correctly (shuffled, stratified or by time)',
      'Recognise the common forms of data leakage and how to prevent them'
    ],
    sections: [
      { type: 'text', title: 'The unglamorous 80%', md: `Before training anything, walk through this checklist:

1. **Look at the data** — a few rows, the column types, summary statistics.
2. **Missing values** — how many per column, and *why* are they missing?
3. **Duplicates** — exact copies, or the same entity recorded twice.
4. **Impossible values** — negative ages, prices of 0, dates in the future.
5. **The label** — is it reliable? Is its distribution balanced?
6. **Split** — set aside validation and test data *before* fitting any preprocessing.

:::analogy Cooking
No chef can rescue spoiled ingredients. Data preparation is checking, washing and chopping the ingredients — dull, essential, and where most kitchen disasters actually begin.
:::` },
      { type: 'code', title: 'A first look with pandas', md: `**pandas** is Python's table library: a \`DataFrame\` is a table with named columns. (The first time you run this, the browser downloads pandas, so it takes a little longer.)`, code: `
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "size_m2":  [120, 80, np.nan, 200, 60, 80],
    "rooms":    [3, 2, 2, 5, 1, 2],
    "city":     ["Lyon", "Paris", "Paris", None, "Lyon", "Paris"],
    "price_k":  [320, 410, 380, 650, 150, 410],
})
print(df.head())                    # first rows
print()
print(df.dtypes)                    # column types
print()
print(df.isna().sum())              # missing values per column
print("duplicate rows:", df.duplicated().sum())
print()
print(df.describe())                # count, mean, std, min, quartiles, max
` },
      { type: 'text', title: 'Missing values', md: `Real datasets have holes. Your options:

| Strategy | When it makes sense | Risk |
| --- | --- | --- |
| **Drop rows** with missing values | few rows affected, missing at random | throws data away; can bias the sample |
| **Drop the column** | mostly missing and not important | loses information |
| **Impute** with the mean / median (numbers) or most frequent value (categories) | moderate missingness | shrinks variance; hides the fact it was missing |
| **Impute + add an "is_missing" indicator column** | missingness itself may be informative | one more feature |
| **Model-based imputation** (predict the missing value) | lots of data, strong relationships | complex; can leak |

The golden rule: **compute imputation statistics on the training set only**, then apply those same numbers to validation and test data. The test set must look like truly unseen data.` },
      { type: 'text', title: 'Training, validation and test sets', md: `You need data the model has never seen to estimate how it will perform in the real world:

- **Training set** (e.g. 70–80%) — used to fit the model's parameters.
- **Validation set** (e.g. 10–15%) — used to choose hyperparameters and compare models.
- **Test set** (e.g. 10–15%) — touched **once**, at the very end, for an honest final estimate.

Why three? Every time you pick something (a model, a learning rate) because it scored best on a dataset, you slightly overfit to that dataset. The test set stays untouched by those decisions.

How to split:
- **Shuffle first** — files are often sorted by date, class or source.
- **Stratify** for classification — keep the class proportions the same in every split (critical for rare classes).
- **Split by time** for anything predicting the future — train on the past, test on later data. Shuffling time series leaks the future into training.
- **Split by group** when examples are related (several photos of the same patient go in the same split).` },
      { type: 'code', title: 'Splitting with NumPy', code: `
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(10, 3))
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])      # sorted by class: shuffling matters!

perm = rng.permutation(len(X))                    # a shuffled order of 0..9
n_test = 3
test_idx, train_idx = perm[:n_test], perm[n_test:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print("train labels:", y_train, " test labels:", y_test)

# Without shuffling, the test set would be the last 3 rows: all class 1!
print("unshuffled test labels:", y[-n_test:])
` },
      { type: 'check', question: { type: 'mcq', q: 'You are predicting next month\'s sales from 5 years of monthly data. How should you split?', options: ['Shuffle all months randomly, then split', 'Train on earlier months, validate and test on the most recent months', 'Use the most recent months for training', 'Split by stratifying on sales'], answer: 1, explain: 'For forecasting, the evaluation must mimic the future: train on the past and test on later periods. Random shuffling would let the model peek at the future.' } },
      { type: 'text', title: 'Data leakage', md: `**Leakage** happens when information that would not be available at prediction time sneaks into training. The model looks brilliant in evaluation and fails in reality.

Common forms:
1. **Preprocessing on all data** — computing the mean for scaling or imputation over the full dataset before splitting. The test set's statistics leak into training.
2. **Target leakage** — a feature that is a consequence of the label. Predicting customer churn with a column \`cancellation_reason\` — it is only filled in *after* the customer churned.
3. **Duplicates across splits** — the same (or nearly the same) example in train and test.
4. **Time leakage** — using future data to predict the past.
5. **Tuning on the test set** — repeatedly checking the test score while making decisions.

:::key The fix is a habit
Split first. Fit every preprocessing step (imputers, scalers, encoders, feature selection) on the **training data only**, then *apply* it to validation and test. scikit-learn's \`Pipeline\` enforces this automatically.
:::

:::warn Too good to be true
If a model scores 99.9% on a hard problem, suspect leakage before celebrating. Check which features matter most — the leak is usually at the top.
:::` },
      { type: 'check', question: { type: 'tf', q: 'It is fine to fill missing values with the mean computed over the whole dataset before splitting into train and test.', answer: false, explain: 'That mean includes test examples, so information from the test set leaks into training. Compute it on the training split only.' } },
      { type: 'text', title: 'Outliers, duplicates and label noise', md: `- **Outliers**: find them with z-scores (|z| > 3) or the IQR rule (below Q1 − 1.5·IQR or above Q3 + 1.5·IQR). Then decide with domain knowledge: a typo (age 999 → remove or fix) or a real extreme (a mansion → keep, maybe log-transform).
- **Duplicates**: drop exact duplicate rows (\`df.drop_duplicates()\`) and think about near-duplicates (the same text with different whitespace).
- **Label noise**: some labels are simply wrong. Spot-check examples, especially those the model gets confidently wrong — they are often mislabelled.` }
    ],
    when: `- Do this preparation **every time**, for every project, before any modelling.
- Use a **random stratified split** for typical classification, a **time-based split** for forecasting or any data with a time order, and a **group split** when examples are related.
- Prefer **median imputation** for skewed numbers and **most-frequent** for categories; add a missing-indicator when missingness may carry signal.
- With small datasets, replace the fixed validation set with **cross-validation** (later lesson).`,
    pitfalls: `- Fitting scalers, imputers or encoders before splitting (leakage).
- Shuffling time series.
- Forgetting to apply exactly the same preprocessing to new data at prediction time.
- Silently dropping rows with missing values and biasing the dataset.
- Evaluating repeatedly on the test set until the number looks good.`,
    takeaways: [
      'Look at the data first: types, missing values, duplicates, impossible values, label balance.',
      'Split into train / validation / test before fitting any preprocessing.',
      'Shuffle (and stratify) for i.i.d. data; split by time for forecasting.',
      'Leakage = information unavailable at prediction time reaching training; it inflates scores.',
      'Fit preprocessing on training data only, then apply it to the other splits.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the main purpose of a separate test set?', options: ['To train the model faster', 'To get an honest estimate of performance on unseen data, untouched by modelling decisions', 'To tune the learning rate', 'To store extra training data'], answer: 1, explain: 'The test set is used once, at the end. Using it for decisions would make it optimistic.' },
      { type: 'multi', q: 'Which of these are data leakage? (select all)', options: ['Scaling features using the mean of the full dataset before splitting', 'Including the column "refund_issued" when predicting whether an order will be returned', 'Imputing missing values with the training-set median', 'Having the same customer\'s duplicate records in both train and test'], answer: [0, 1, 3], explain: 'Training-set statistics are correct practice. The others let test or future information reach training.' },
      { type: 'mcq', q: 'A classification dataset has 2% positive examples. Why stratify the split?', options: ['To make the classes balanced 50/50', 'So each split keeps about 2% positives — otherwise a small test set might contain almost none', 'Stratification speeds up training', 'It removes outliers'], answer: 1, explain: 'Stratification preserves class proportions in every split so rare classes are represented.' },
      { type: 'tf', q: 'For data with a natural time order, a random shuffled split can make results look better than they will be in production.', answer: true, explain: 'Shuffling lets the model train on data from after the test period — future information leaks into training.' },
      { type: 'mcq', q: 'A model scores 99.8% accuracy on a problem experts find hard. What should you check first?', options: ['Whether to add more layers', 'Whether some feature leaks the label or test examples duplicate training ones', 'Whether to use a bigger test set', 'Nothing — ship it'], answer: 1, explain: 'Implausibly good results are usually leakage. Inspect the most important features and look for duplicates.' }
    ],
    problems: [
      {
        id: 'train-test-split', title: 'Shuffled Train/Test Split', difficulty: 'easy',
        prompt: `Implement \`train_test_split(X, y, test_ratio=0.2, seed=0)\` with this **exact** procedure (so results are reproducible and can be checked):

1. \`n = len(X)\`, \`rng = np.random.default_rng(seed)\`, \`perm = rng.permutation(n)\`.
2. \`n_test = int(round(n * test_ratio))\`.
3. The **first** \`n_test\` indices of \`perm\` form the test set, the rest the training set.
4. Return \`X_train, X_test, y_train, y_test\` (NumPy arrays), with X and y rows kept aligned.

\`\`\`text
X = np.arange(10).reshape(5, 2); y = np.arange(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.4, seed=1)
# -> 3 training rows and 2 test rows, chosen by the seeded permutation
\`\`\``,
        starter: `
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=0):
    """Shuffle with a seeded permutation, then split off the first n_test indices as the test set."""
    X = np.asarray(X)
    y = np.asarray(y)
    # TODO
    pass
`,
        solution: `
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
`,
        hints: ['`perm[:n_test]` are the test indices and `perm[n_test:]` the training indices.', 'Index X and y with the **same** index arrays so rows stay aligned: `X[train_idx]`, `y[train_idx]`.'],
        explain: `Shuffling *indices* rather than the arrays themselves keeps X and y perfectly aligned. Seeding the generator makes the split reproducible — essential when you compare models, because they must be evaluated on the same test examples. scikit-learn's \`train_test_split\` does the same thing (plus stratification).`,
        tests: [
          { name: 'seeded split', setup: 'X = np.arange(10).reshape(5, 2)\ny = np.arange(5)', call: 'train_test_split(X, y, 0.4, seed=1)' },
          { name: 'sizes and alignment', setup: 'X = np.arange(40).reshape(20, 2)\ny = np.arange(20) * 2', code: 'Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=3)\nassert len(Xte) == 5 and len(Xtr) == 15, "expected 15 train / 5 test rows"\nassert np.all(Xtr[:, 0] == ytr), "X and y rows must stay aligned"\nassert np.all(Xte[:, 0] == yte)\nassert set(ytr.tolist()).isdisjoint(set(yte.tolist())), "train and test must not overlap"' },
          { name: 'different seed, different split', setup: 'X = np.arange(30).reshape(15, 2)\ny = np.arange(15)', call: 'train_test_split(X, y, 0.2, seed=7)', hidden: true },
          { name: 'rounding of n_test', setup: 'X = np.arange(7)\ny = np.arange(7)', code: 'Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed=0)\nassert len(Xte) == 2, "round(7 * 0.3) = 2 test rows"', hidden: true }
        ]
      },
      {
        id: 'impute-mean', title: 'Leak-Free Mean Imputation', difficulty: 'easy',
        prompt: `Implement \`impute_mean(X_train, X_test)\` for 2-D float arrays that contain \`np.nan\` for missing values:

1. Compute each column's mean **from the training data only**, ignoring NaNs (\`np.nanmean(X_train, axis=0)\`).
2. Return **new** arrays \`(train_filled, test_filled)\` where every NaN in either array is replaced by its column's **training** mean.
3. Do not modify the inputs.

\`\`\`text
X_train = [[1, nan], [3, 4]]      column means: [2, 4]
X_test  = [[nan, 10]]
-> train_filled = [[1, 4], [3, 4]],  test_filled = [[2, 10]]
\`\`\``,
        starter: `
import numpy as np

def impute_mean(X_train, X_test):
    """Fill NaNs in both arrays with the per-column means of X_train."""
    # TODO: np.nanmean, np.isnan and np.where will help
    pass
`,
        solution: `
import numpy as np

def impute_mean(X_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    means = np.nanmean(X_train, axis=0)                     # learned from training data only
    train_filled = np.where(np.isnan(X_train), means, X_train)
    test_filled = np.where(np.isnan(X_test), means, X_test)
    return train_filled, test_filled
`,
        hints: ['`np.nanmean(X_train, axis=0)` gives one mean per column, skipping NaNs.', '`np.where(np.isnan(X), means, X)` broadcasts the (d,) means across every row.'],
        explain: `This is the fit/transform pattern in miniature: **fit** (compute the means) on training data, **transform** both splits with those same numbers. Using the test set's own means would be leakage — at prediction time a single new example has no "column mean" of its own.`,
        tests: [
          { call: 'impute_mean(np.array([[1.0, np.nan], [3.0, 4.0]]), np.array([[np.nan, 10.0]]))', expect: '(np.array([[1.0, 4.0], [3.0, 4.0]]), np.array([[2.0, 10.0]]))' },
          { name: 'test uses training means (no leakage)', call: 'impute_mean(np.array([[0.0], [2.0]]), np.array([[np.nan], [np.nan], [100.0]]))', expect: '(np.array([[0.0], [2.0]]), np.array([[1.0], [1.0], [100.0]]))' },
          { name: 'inputs not modified', code: 'tr = np.array([[1.0, np.nan], [3.0, 4.0]])\nte = np.array([[np.nan, 1.0]])\nimpute_mean(tr, te)\nassert np.isnan(tr[0, 1]) and np.isnan(te[0, 0]), "impute_mean must not modify its inputs"', hidden: true },
          { name: 'random data with holes', setup: 'rng = np.random.default_rng(5)\ntr = rng.normal(size=(30, 4)); tr[rng.random((30, 4)) < 0.2] = np.nan\nte = rng.normal(size=(10, 4)); te[rng.random((10, 4)) < 0.3] = np.nan', call: 'impute_mean(tr, te)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'pandas — 10 minutes to pandas', url: 'https://pandas.pydata.org/docs/user_guide/10min.html' },
      { title: 'scikit-learn — Common pitfalls and recommended practices (data leakage)', url: 'https://scikit-learn.org/stable/common_pitfalls.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'feature-engineering', branch: 'code', title: 'Feature Engineering & Scaling', short: 'Feature Engineering', level: 2, minutes: 40, prereqs: ['data-preparation'],
    tags: ['scaling', 'standardization', 'one-hot', 'encoding', 'features'],
    summary: 'Making features a model can use: scaling numbers, encoding categories, transforming skewed values and creating informative new features.',
    why: `Models only see numbers, and *how* you present information changes what they can learn. A distance-based model will ignore "number of bedrooms" if "size in m²" is a thousand times larger; a linear model cannot use "city" until it is encoded; a price that spans three orders of magnitude is easier to model on a log scale; and "price per m²" can be far more informative than price and size separately. On tabular data, thoughtful features often beat fancier models — and even in deep learning, the same ideas appear as normalization layers and embeddings.`,
    objectives: [
      'Standardize and min-max scale features with the fit/transform pattern',
      'Know which models need feature scaling and which do not',
      'Encode categorical variables with one-hot or ordinal encoding and handle unseen categories',
      'Transform skewed features and create useful derived features'
    ],
    sections: [
      { type: 'text', title: 'Why the representation matters', md: `Imagine predicting house prices from \`size_m2\` (50–400) and \`bedrooms\` (1–6). Two common problems:

1. **Scale.** In a distance computation, a 10 m² difference outweighs a 4-bedroom difference — purely because of units. Gradient descent also struggles when features differ hugely in scale: the loss surface becomes a long narrow valley.
2. **Type.** A model can't multiply \`"Lyon"\` by a weight. Categories must become numbers — without inventing a fake order.

Feature engineering is the craft of choosing representations that make the true patterns easy to learn.` },
      { type: 'text', title: 'Scaling numeric features', md: `| Method | Formula | Result |
| --- | --- | --- |
| **Standardization** (z-score) | (x − mean) / std | mean 0, std 1; the default choice |
| **Min-max scaling** | (x − min) / (max − min) | values in [0, 1]; sensitive to outliers |
| **Robust scaling** | (x − median) / IQR | resists outliers |

**Which models care?**
- **Need scaling:** anything using distances or dot products with regularization — k-nearest neighbours, k-means, SVMs, PCA, linear and logistic regression with regularization, and neural networks (training is much faster).
- **Don't need it:** decision trees, random forests and gradient-boosted trees — they split on thresholds, which are unaffected by monotonic rescaling.

As always: **fit** the scaler (learn mean and std) on training data, then **transform** all splits with those numbers.` },
      { type: 'code', title: 'A scaler with the fit / transform pattern', md: `scikit-learn's preprocessing objects all follow this two-step API. Here is a minimal standard scaler from scratch.`, code: `
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)   # constant column: avoid divide-by-zero
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

X_train = np.array([[120, 3], [80, 2], [200, 5], [60, 1]], dtype=float)
X_test  = np.array([[100, 2], [300, 6]], dtype=float)

scaler = StandardScaler().fit(X_train)      # learn statistics from TRAINING data only
print("means:", scaler.mean_, " stds:", scaler.scale_.round(2))
print("train scaled:")
print(scaler.transform(X_train).round(2))
print("test scaled (uses the training statistics):")
print(scaler.transform(X_test).round(2))
` },
      { type: 'text', title: 'Encoding categorical features', md: `**Nominal** categories have no order (city, colour, product type). Encoding them as 0, 1, 2 would tell a linear model that "Paris" (2) is twice "Lyon" (1) — nonsense. Use **one-hot encoding**: one column per category, with a 1 in the column that applies.

| city | is_Lyon | is_Paris | is_Nice |
| --- | --- | --- | --- |
| Paris | 0 | 1 | 0 |
| Nice | 0 | 0 | 1 |

**Ordinal** categories *do* have an order (small < medium < large, or education levels) — mapping them to 0, 1, 2 is reasonable.

Practical issues:
- **Unseen categories** at prediction time: map them to an all-zeros row (or an explicit "other" column).
- **High cardinality** (thousands of zip codes or product ids): one-hot becomes huge. Alternatives: group rare categories into "other", hashing, target encoding (carefully, it leaks easily), or **learned embeddings** — dense vectors per category, exactly how LLMs represent words.` },
      { type: 'check', question: { type: 'mcq', q: 'Why is encoding colours as red=0, green=1, blue=2 a problem for a linear model?', options: ['Numbers use more memory than strings', 'It invents an order and distances (blue = 2 × green) that don\'t exist', 'Linear models cannot use integers', 'It is fine; there is no problem'], answer: 1, explain: 'A linear model would treat the codes as quantities. One-hot encoding avoids inventing an order.' } },
      { type: 'code', title: 'One-hot encoding by hand', code: `
import numpy as np

cities = ["Paris", "Nice", "Lyon", "Paris", "Marseille"]   # "Marseille" never seen in training
categories = ["Lyon", "Nice", "Paris"]                     # learned from the training data

index = {c: i for i, c in enumerate(categories)}
encoded = np.zeros((len(cities), len(categories)))
for row, city in enumerate(cities):
    if city in index:                                      # unseen category -> all zeros
        encoded[row, index[city]] = 1.0
print(categories)
print(encoded)
` },
      { type: 'text', title: 'Transforming and creating features', md: `- **Log transforms** for skewed, positive quantities (incomes, prices, counts): \`np.log1p(x)\`. They compress the long tail and often make relationships more linear.
- **Ratios and differences** that match the domain: price per m², debt-to-income, time since last purchase.
- **Date and time parts**: hour of day, day of week, month, "is weekend". Cyclical features can be encoded with sine/cosine so that 23:00 is close to 00:00.
- **Binning**: turning age into age groups when the effect is step-like.
- **Interactions and polynomials**: x₁·x₂ or x² let linear models fit curves — but add too many and you overfit (see *Overfitting*).
- **Text** becomes numbers via bag-of-words, TF-IDF or embeddings — the start of the NLP branch.

Always ask: *would this feature be available at prediction time?* — a feature computed from the future is leakage.` },
      { type: 'check', question: { type: 'mcq', q: 'Which model is essentially unaffected by standardizing its features?', options: ['k-nearest neighbours', 'A decision tree', 'Logistic regression with L2 regularization', 'A neural network'], answer: 1, explain: 'Trees choose thresholds on one feature at a time, so rescaling a feature just rescales its thresholds.' } }
    ],
    when: `- **Standardize** by default for linear models, SVMs, kNN, PCA, k-means and neural networks; skip scaling for tree ensembles.
- **One-hot** low-cardinality nominal features; **ordinal-encode** naturally ordered ones; consider **embeddings** for high-cardinality ids when you have lots of data.
- **Log-transform** heavily right-skewed positive features.
- Add **domain ratios and time features** when domain knowledge suggests them — and verify each helps on validation data.`,
    pitfalls: `- Fitting scalers or encoders on all data (leakage) — fit on training data only.
- Forgetting to save the fitted scaler/encoder with the model; production data must be transformed identically.
- Integer-encoding nominal categories for linear models or neural nets.
- Exploding the feature count with one-hot encodings of IDs.
- Creating features that use information from the future.`,
    takeaways: [
      'Distance- and gradient-based models need scaled features; tree models do not.',
      'Standardization: (x − mean) / std, with statistics learned on training data only.',
      'One-hot encode nominal categories; map unseen categories to all zeros.',
      'Log-transform skewed values; build ratios, time parts and interactions from domain knowledge.',
      'Every transformation must be fitted on training data and reapplied identically at prediction time.'
    ],
    quiz: [
      { type: 'fill', q: 'A feature has training mean 50 and standard deviation 10. What is the standardized value of 65?', answer: ['1.5'], explain: '(65 − 50) / 10 = 1.5.' },
      { type: 'mcq', q: 'One-hot encoding a "country" column with 5 distinct countries produces how many columns?', options: ['1', '5', '2', '25'], answer: 1, explain: 'One indicator column per category: 5.' },
      { type: 'mcq', q: 'Incomes in a dataset range from 10k to 10 million with a long right tail. A common, helpful transform is…', options: ['Squaring the values', 'Taking the logarithm', 'One-hot encoding them', 'Removing all values above the mean'], answer: 1, explain: 'A log transform compresses the tail so that ratios (doubling) become equal steps.' },
      { type: 'tf', q: 'At prediction time you should re-fit the scaler on the new incoming data before transforming it.', answer: false, explain: 'Use the scaler fitted on the training data — the model was trained on features scaled with those statistics.' },
      { type: 'multi', q: 'Which models usually benefit from feature scaling? (select all)', options: ['k-nearest neighbours', 'Random forest', 'Neural network', 'Support vector machine'], answer: [0, 2, 3], explain: 'Distance and gradient-based models are sensitive to scale; tree ensembles are not.' }
    ],
    problems: [
      {
        id: 'standard-scaler', title: 'Build a StandardScaler', difficulty: 'easy',
        prompt: `Implement a \`StandardScaler\` class with the scikit-learn API:

- \`fit(X)\` — learn \`self.mean_\` (per-column mean) and \`self.scale_\` (per-column population std, \`ddof=0\`). If a column's std is 0, use 1 instead. Return \`self\`.
- \`transform(X)\` — return \`(X - mean_) / scale_\` as a new float array.
- \`fit_transform(X)\` — fit, then transform the same data.

\`\`\`text
sc = StandardScaler().fit(np.array([[1.0, 10.0], [3.0, 10.0]]))
sc.mean_   ->  [2.0, 10.0]
sc.scale_  ->  [1.0, 1.0]      (second column is constant, so its scale is 1)
sc.transform(np.array([[5.0, 12.0]]))  ->  [[3.0, 2.0]]
\`\`\``,
        starter: `
import numpy as np

class StandardScaler:
    def fit(self, X):
        """Learn per-column mean_ and scale_ (std, with 0 replaced by 1). Return self."""
        # TODO
        return self

    def transform(self, X):
        """Standardize X using the learned statistics."""
        # TODO
        pass

    def fit_transform(self, X):
        # TODO
        pass
`,
        solution: `
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
`,
        hints: ['`X.mean(axis=0)` and `X.std(axis=0)` give per-column statistics.', '`np.where(std == 0, 1.0, std)` replaces zero stds.', '`fit_transform` can simply be `return self.fit(X).transform(X)`.'],
        explain: `Storing learned statistics on the object (with a trailing underscore, as scikit-learn does) is what lets you apply the *same* transformation to test data and to live data later. Returning \`self\` from \`fit\` enables chaining like \`StandardScaler().fit(X).transform(X)\`.`,
        tests: [
          { call: 'StandardScaler().fit(np.array([[1.0, 10.0], [3.0, 10.0]])).mean_', expect: 'np.array([2.0, 10.0])' },
          { call: 'StandardScaler().fit(np.array([[1.0, 10.0], [3.0, 10.0]])).scale_', expect: 'np.array([1.0, 1.0])' },
          { call: 'StandardScaler().fit(np.array([[1.0, 10.0], [3.0, 10.0]])).transform(np.array([[5.0, 12.0]]))', expect: 'np.array([[3.0, 2.0]])' },
          { name: 'fit_transform gives mean 0 and std 1', setup: 'rng = np.random.default_rng(4)\nX = rng.normal(7, 3, size=(100, 5))', code: 'Z = StandardScaler().fit_transform(X)\nassert np.allclose(Z.mean(axis=0), 0, atol=1e-9)\nassert np.allclose(Z.std(axis=0), 1)', hidden: true },
          { name: 'test data uses training statistics', setup: 'rng = np.random.default_rng(9)\nXtr = rng.normal(2, 5, size=(40, 3))\nXte = rng.normal(-1, 1, size=(6, 3))', call: 'StandardScaler().fit(Xtr).transform(Xte)', hidden: true }
        ]
      },
      {
        id: 'one-hot-encode', title: 'One-Hot Encoding with Unknown Categories', difficulty: 'easy',
        prompt: `Implement \`one_hot(values, categories)\`:

- \`values\` is a list of category labels (strings or numbers).
- \`categories\` is the list of known categories, in column order.
- Return a NumPy float array of shape \`(len(values), len(categories))\` with a 1.0 in the column of each value's category and 0.0 elsewhere.
- A value that is **not** in \`categories\` gets an all-zero row.

\`\`\`text
one_hot(["b", "a", "z"], ["a", "b", "c"])
-> [[0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 0.]]
\`\`\``,
        starter: `
import numpy as np

def one_hot(values, categories):
    """(n, k) one-hot matrix; unknown values give all-zero rows."""
    # TODO: build a dict category -> column index, then fill a zeros array
    pass
`,
        solution: `
import numpy as np

def one_hot(values, categories):
    index = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(values), len(categories)))
    for row, v in enumerate(values):
        col = index.get(v)
        if col is not None:
            out[row, col] = 1.0
    return out
`,
        hints: ['Make a lookup dict: `index = {c: i for i, c in enumerate(categories)}`.', 'Start from `np.zeros((len(values), len(categories)))` and set one entry per known value.'],
        explain: `A dictionary lookup gives O(1) access to each category's column; \`index.get(v)\` returns None for unseen values, which we leave as zeros. This is the same lookup a tokenizer does from token to id — and an embedding layer is equivalent to multiplying a one-hot matrix by a weight matrix.`,
        tests: [
          { call: 'one_hot(["b", "a", "z"], ["a", "b", "c"])', expect: 'np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])' },
          { call: 'one_hot([2, 2, 0], [0, 1, 2])', expect: 'np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])' },
          { name: 'shape with no values', code: 'out = one_hot([], ["x", "y"])\nassert out.shape == (0, 2), "expected shape (0, 2)"', hidden: true },
          { name: 'every known row sums to 1', setup: 'rng = np.random.default_rng(1)\ncats = ["red", "green", "blue", "black"]\nvals = [cats[i] for i in rng.integers(0, 4, size=30)]', code: 'M = one_hot(vals, cats)\nassert M.shape == (30, 4)\nassert np.allclose(M.sum(axis=1), 1)\nassert all(cats[int(np.argmax(r))] == v for r, v in zip(M, vals))', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn — Preprocessing data', url: 'https://scikit-learn.org/stable/modules/preprocessing.html' },
      { title: "Kuhn & Johnson — Feature Engineering and Selection (free book)", url: "https://bookdown.org/max/FES/" }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
