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
      { type: 'text', title: "Exceptions: when things go wrong", md: `When Python cannot do what you asked, it **raises an exception** and, unless something catches it, stops with a traceback. The ones you will meet most often:

| Exception | Typical cause |
| --- | --- |
| \`KeyError\` | a dictionary has no such key: \`config["batch_size"]\` |
| \`IndexError\` | a list index past the end |
| \`TypeError\` | the wrong kind of value: \`"3" + 4\`, or calling something that is not a function |
| \`ValueError\` | the right type but an unacceptable value: \`int("abc")\`, mismatched shapes in many libraries |
| \`ZeroDivisionError\` | dividing by zero with plain Python numbers |
| \`FileNotFoundError\` | a path that does not exist |

\`try\` / \`except\` handles a problem you expect; \`raise\` signals one in your own code:

\`\`\`python
try:
    batch = config["batch_size"]
except KeyError:
    batch = 32                     # fall back to a default
\`\`\`

Catch only the exceptions you expect and know how to handle. A bare \`except:\` that swallows everything also hides real bugs, such as a typo in a variable name. For missing dictionary keys, \`config.get("batch_size", 32)\` is shorter still.

**Assertions** document assumptions and fail loudly when they break: \`assert len(X) == len(y), "one label per row"\`. They are cheap insurance in data code, where a silently mismatched array is far worse than a crash.` },
      { type: 'text', title: "Names, references and copies", md: `A Python variable is a **name attached to an object**, not a box holding a value. \`b = a\` attaches a second name to the same object; nothing is copied. That matters for **mutable** objects (lists, dicts, sets, NumPy arrays), whose contents can change in place:

- after \`b = a\`, \`b.append(4)\` changes the list that \`a\` also names;
- \`list(a)\`, \`a.copy()\` or \`dict(d)\` make a **shallow copy**: a new outer container holding the same items; \`copy.deepcopy\` copies nested structures all the way down.

Immutable objects (numbers, strings, tuples) cannot change in place, so sharing them is always safe: \`x += 1\` simply makes \`x\` name a new number.

The classic trap is a **mutable default argument**. Default values are evaluated once, when the function is defined, so every call that relies on the default shares the same list. Use \`None\` and create the list inside:

\`\`\`python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
\`\`\`` },
      { type: 'code', title: "Exceptions, aliases and the mutable-default trap", code: `config = {"lr": 0.01, "epochs": 5}
try:
    batch = config["batch_size"]
except KeyError as err:
    print("missing key:", err)
print("with a default:", config.get("batch_size", 32))

def safe_ratio(a, b):
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b

try:
    safe_ratio(1, 0)
except ValueError as err:
    print("caught:", err)

a = [1, 2, 3]
b = a                          # a second name for the same list
b.append(4)
print("a after changing b:", a)
c = list(a)                    # a real (shallow) copy
c.append(5)
print("a after changing the copy:", a)

def add_item(item, items=[]):  # the default list is created only once!
    items.append(item)
    return items

add_item(1)
print("second call returns:", add_item(2))`, after: `The second call returns [1, 2], not [2]: both calls appended to the single default list that was created when the function was defined. With \`items=None\` and a fresh list inside the function, every call starts empty.`, predict: {"type":"mcq","q":"add_item(1) is called, then add_item(2). What does the second call return?","options":["[2]","[1, 2]","An error","[]"],"answer":1,"explain":"The default list is created once, when the function is defined, and every call that uses the default appends to that same list.","expect":["second call returns: [1, 2]"]} },
      { type: 'text', title: "Iterators, generators and batching", md: `A \`for\` loop works on anything **iterable**: lists, strings, dicts (their keys), files (their lines), \`range\`, and **generators**. A generator function uses \`yield\` instead of \`return\`: each \`yield\` hands out one value and pauses the function until the next value is requested. Nothing is computed in advance, so a generator can describe a stream far larger than memory.

\`\`\`python
def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
\`\`\`

This is exactly how training data reaches a model: a data loader is an iterator that yields one mini-batch at a time, reading, decoding and augmenting examples only when they are needed. A **generator expression** such as \`(x * x for x in data)\` is the lazy cousin of a list comprehension.

Files work the same way. \`with open(path) as f:\` opens a file and guarantees it is closed again, even if an error occurs; looping over \`f\` yields one line at a time. For structured settings, the \`json\` module converts between Python dicts and JSON text: \`json.dump(config, f)\` saves an experiment's configuration, and \`json.load(f)\` reads it back.` },
      { type: 'code', title: "Batches, generators and a saved config", code: `import json
import os
import tempfile

def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]          # hand out one batch, then pause

for i, batch in enumerate(batches(list(range(10)), 4)):
    print("batch", i, batch)

squares = (x * x for x in range(1_000_000))     # a generator expression: nothing computed yet
print("sum of a million squares:", sum(squares))

config = {"model": "tiny-gpt", "lr": 0.001, "layers": [64, 64]}
path = os.path.join(tempfile.gettempdir(), "config.json")
with open(path, "w") as f:                      # 'with' closes the file even if an error occurs
    json.dump(config, f)
with open(path) as f:
    loaded = json.load(f)
print("round trip ok:", loaded == config, loaded)`, after: `The last batch has only 2 items, because 10 is not a multiple of 4; that is why data loaders have a \`drop_last\` option. The generator expression added up a million squares without ever building a list of a million numbers.` },
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
      { type: 'mcq', q: 'Why do ML libraries wrap models in classes?', options: ['Classes run faster than functions', 'A model needs to keep its parameters (data) together with its methods like predict/forward', 'Python requires it', 'To avoid using dictionaries'], answer: 1, explain: 'The parameters live on the object (self.w) and the methods use them — the PyTorch nn.Module pattern.' },
      {"type":"mcq","q":"After a = [1, 2]; b = a; b.append(3), what is a?","options":["[1, 2]","[1, 2, 3]","[3]","An error"],"answer":1,"explain":"b = a makes b a second name for the same list, so appending through b changes the list that a names too."},
      {"type":"mcq","q":"Which exception does config['missing'] raise when the dict has no such key?","options":["IndexError","ValueError","KeyError","TypeError"],"answer":2,"explain":"Looking up a missing dictionary key raises KeyError; config.get('missing', default) returns the default instead."},
      {"type":"tf","q":"A generator computes all of its values as soon as it is created.","answer":false,"explain":"A generator is lazy: it computes each value only when the next one is requested, which is why it can describe streams larger than memory."}
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
      { type: 'check', question: { type: 'fill', q: 'What is the shape of `np.ones((3, 1)) + np.ones(4)`? (write it like (a, b))', answer: ['(3, 4)', '(3,4)'], display: '(3, 4)', explain: 'Align from the right: (3, 1) vs (·, 4). 1 stretches to 4 and the missing dimension stretches to 3 → (3, 4).' } },
      { type: 'text', title: "dtypes, precision and memory", md: `Every array has one **dtype**, which decides both how much memory each element takes and which numbers it can represent:

| dtype | Bytes | Approximate range | Significant decimal digits | Typical use |
| --- | --- | --- | --- | --- |
| float64 | 8 | ±1.8 × 10³⁰⁸ | about 16 | NumPy's default; statistics and scientific computing |
| float32 | 4 | ±3.4 × 10³⁸ | about 7 | the classic deep-learning default |
| bfloat16 | 2 | ±3.4 × 10³⁸ | about 2 to 3 | training large models on modern accelerators |
| float16 | 2 | ±65,504 | about 3 | inference, and mixed-precision training with loss scaling |
| int8 | 1 | −128 to 127 | exact integers | quantised model weights |
| int64 | 8 | ±9.2 × 10¹⁸ | exact integers | counts, indices, token ids |

Floating-point numbers keep a fixed number of significant bits, so the **gap** between neighbouring representable numbers grows with their size, and the result of every operation is rounded to the nearest representable value. Three consequences show up again and again:
- **Comparisons need a tolerance.** 0.1 + 0.2 is not exactly 0.3 in binary floating point, so compare results with \`np.isclose\`, never with \`==\`.
- **Large plus small loses the small.** In float32, 16,777,216 + 1 is still 16,777,216. Adding many small values to a large total silently drops information, which is why libraries accumulate sums in higher precision or in pairs.
- **Overflow and wrap-around.** float16 turns anything above 65,504 into \`inf\`, and integer arrays silently wrap around (in int8, 127 + 1 = −128). NaN ("not a number") comes from 0/0 or inf − inf, spreads through every computation it touches and is not even equal to itself: test for it with \`np.isnan\`.

Memory is simply elements × bytes: a million float64 values take 8 MB and the same values in float16 take 2 MB. A model with 7 billion parameters needs 28 GB in float32, 14 GB in bfloat16 and about 3.5 GB at 4 bits per weight, which is the whole point of [quantization](#/concept/quantization).` },
      { type: 'chart', title: "The gap between neighbouring floating-point numbers", after: `Every format keeps a fixed number of significant bits, so the gap grows with the size of the number. Near 1,000, float16 can only step in units of 0.5 and bfloat16 in units of 4; float32 resolves about 0.00006, and float64 about 10⁻¹³. float16 also stops at 65,504 (larger values become inf), while bfloat16 keeps float32's huge range by sacrificing precision. That trade-off is why it is the favourite format for training large models.`, chart: {"kind":"line","x":[0.001,0.00178,0.00316,0.00562,0.01,0.0178,0.0316,0.0562,0.1,0.178,0.316,0.562,1,1.78,3.16,5.62,10,17.8,31.6,56.2,100,178,316,562,1000,1780,3160,5620,10000,17800,31600,56200,100000,178000,316000,562000,1000000],"xLog":true,"yLog":true,"points":false,"series":[{"name":"float16","y":[9.537e-7,9.537e-7,0.000001907,0.000003815,0.000007629,0.00001526,0.00003052,0.00003052,0.00006104,0.0001221,0.0002441,0.0004883,0.0009766,0.0009766,0.001953,0.003906,0.007812,0.01562,0.01562,0.03125,0.0625,0.125,0.25,0.5,0.5,1,2,4,8,16,16,32,null,null,null,null,null]},{"name":"bfloat16","y":[0.000007629,0.000007629,0.00001526,0.00003052,0.00006104,0.0001221,0.0002441,0.0002441,0.0004883,0.0009766,0.001953,0.003906,0.007812,0.007812,0.01562,0.03125,0.0625,0.125,0.125,0.25,0.5,1,2,4,4,8,16,32,64,128,128,256,512,1024,2048,4096,4096]},{"name":"float32","y":[1.164e-10,1.164e-10,2.328e-10,4.657e-10,9.313e-10,1.863e-9,3.725e-9,3.725e-9,7.451e-9,1.49e-8,2.98e-8,5.96e-8,1.192e-7,1.192e-7,2.384e-7,4.768e-7,9.537e-7,0.000001907,0.000001907,0.000003815,0.000007629,0.00001526,0.00003052,0.00006104,0.00006104,0.0001221,0.0002441,0.0004883,0.0009766,0.001953,0.001953,0.003906,0.007812,0.01562,0.03125,0.0625,0.0625]},{"name":"float64","y":[2.168e-19,2.168e-19,4.337e-19,8.674e-19,1.735e-18,3.469e-18,6.939e-18,6.939e-18,1.388e-17,2.776e-17,5.551e-17,1.11e-16,2.22e-16,2.22e-16,4.441e-16,8.882e-16,1.776e-15,3.553e-15,3.553e-15,7.105e-15,1.421e-14,2.842e-14,5.684e-14,1.137e-13,1.137e-13,2.274e-13,4.547e-13,9.095e-13,1.819e-12,3.638e-12,3.638e-12,7.276e-12,1.455e-11,2.91e-11,5.821e-11,1.164e-10,1.164e-10]}],"xLabel":"size of the number (log scale)","yLabel":"gap to the next representable number"} },
      { type: 'code', title: "Precision surprises", code: `import numpy as np
big = np.float32(16_777_216)                      # 2**24
print(f"float32: {float(big + np.float32(1)):,.0f}    float64: {float(np.float64(16_777_216) + 1):,.0f}")
print("float32 lost the +1:", bool(big + np.float32(1) == big))

print(f"largest float16: {float(np.finfo(np.float16).max):,.0f}")
with np.errstate(over="ignore"):
    print("float16(70000) becomes:", np.float16(70000))
print("int8 wraps around:", np.array([120, 127], dtype=np.int8) + np.int8(10))

print("0.1 + 0.2 == 0.3:", 0.1 + 0.2 == 0.3, "   np.isclose(0.1 + 0.2, 0.3):", np.isclose(0.1 + 0.2, 0.3))
print("nan == nan:", np.nan == np.nan, "   np.isnan(nan):", np.isnan(np.nan))

values = np.ones(1_000_000)
for dt in ["float64", "float32", "float16", "int8"]:
    print(f"one million {dt:<8} values: {values.astype(dt).nbytes / 1e6:.0f} MB")`, predict: {"type":"mcq","q":"In float32, what is 16,777,216 + 1?","options":["16,777,217","16,777,216","inf","An error"],"answer":1,"explain":"float32 has 24 significant bits. Above 2²⁴ = 16,777,216 the gap between neighbouring floats is 2, so adding 1 rounds back to the same number.","expect":["float32 lost the +1: True"]} },
      { type: 'text', title: "Reshaping, stacking and everyday moves", md: `| Operation | What it does | Example shapes |
| --- | --- | --- |
| \`X.reshape(6, -1)\` | the same data in a new shape; \`-1\` means "work it out" | (2, 3, 4) → (6, 4) |
| \`X.ravel()\` or \`X.reshape(-1)\` | flatten to one dimension | (2, 3, 4) → (24,) |
| \`X.T\`, \`X.transpose(0, 2, 1)\` | swap axes | (2, 3, 4) → (2, 4, 3) |
| \`np.concatenate([a, b], axis=0)\` | join along an existing axis | (2, 3) and (2, 3) → (4, 3) |
| \`np.stack([a, b])\` | join along a **new** axis | (2, 3) and (2, 3) → (2, 2, 3) |
| \`v[:, None]\`, \`v[None, :]\` | add an axis of size 1 (\`None\` is \`np.newaxis\`) | (3,) → (3, 1) or (1, 3) |
| \`np.where(cond, a, b)\` | element-wise if/else | |
| \`np.eye(k)[labels]\` | one-hot encode integer labels | (n,) → (n, k) |

Reshaping never changes the order of the elements in memory (row by row, last axis fastest); it only regroups them. Transposing changes the order in which you walk through them. Mixing the two up is a classic source of silently scrambled images and batches: to turn a batch of images from (batch, height, width, channels) into (batch, channels, height, width) you need \`transpose\`, not \`reshape\`.

One more recipe you will need: the **numerically stable softmax**. exp(1000) overflows to inf, so a naive softmax of large logits returns NaN. Subtracting the largest logit first changes nothing mathematically, because the common factor cancels between numerator and denominator, but it keeps every exponent at or below 0.` },
      { type: 'code', title: "Reshape, stack, one-hot and a stable softmax", code: `import numpy as np
X = np.arange(24).reshape(2, 3, 4)
print(X.shape, X.reshape(6, -1).shape, X.reshape(-1).shape, X.transpose(0, 2, 1).shape)

a, b = np.ones((2, 3)), np.zeros((2, 3))
print(np.concatenate([a, b]).shape, np.stack([a, b]).shape, np.concatenate([a, b], axis=1).shape)

v = np.array([1.0, 2.0, 3.0])
print(v[:, None].shape, v[None, :].shape)
print(np.where(v > 1.5, "big", "small"))

labels = np.array([2, 0, 1, 2])
print(np.eye(3)[labels])                          # one row per label, a 1 in that label's column

def softmax_naive(z):
    return np.exp(z) / np.exp(z).sum()

def softmax(z):
    e = np.exp(z - z.max())                       # the largest exponent becomes exp(0) = 1
    return e / e.sum()

z = np.array([1000.0, 1001.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive softmax: ", softmax_naive(z))
print("stable softmax:", softmax(z).round(4))`, predict: {"type":"mcq","q":"X has shape (2, 3, 4). What is the shape of X.reshape(6, -1)?","options":["(6, 4)","(6, 24)","(6, 2)","An error"],"answer":0,"explain":"X has 24 elements; with 6 rows, -1 is worked out as 24 / 6 = 4.","expect":["(2, 3, 4) (6, 4) (24,) (2, 4, 3)"]} }
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
      { type: 'mcq', q: 'Why is `a @ b` so much faster than a Python for-loop computing the same dot product?', options: ['It uses a different algorithm with fewer multiplications', 'It runs in compiled code over contiguous typed memory, avoiding Python\'s per-element overhead', 'It skips small numbers', 'It caches the result'], answer: 1, explain: 'Same arithmetic, but no interpreter overhead per element, plus CPU vector instructions.' },
      {"type":"mcq","q":"Why should you compare floating-point results with np.isclose instead of ==?","options":["== is slower","Rounding makes mathematically equal results differ in their last bits","np.isclose also works on strings","== does not work on arrays"],"answer":1,"explain":"Every floating-point operation rounds, so 0.1 + 0.2 and 0.3 differ in the last bit. np.isclose compares within a small tolerance."},
      {"type":"fill","q":"How many megabytes does an array of 2 million float32 values take?","answer":["8","8 MB","8MB","8.0"],"explain":"float32 uses 4 bytes per value: 2,000,000 × 4 = 8,000,000 bytes = 8 MB."},
      {"type":"mcq","q":"a and b both have shape (2, 3). What is the shape of np.stack([a, b])?","options":["(4, 3)","(2, 6)","(2, 2, 3)","(2, 3, 2)"],"answer":2,"explain":"np.stack joins along a new first axis; np.concatenate would give (4, 3)."}
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
      { type: 'code', title: "Handling missing data in pandas", md: `pandas marks missing values as **NaN** ("not a number"; \`None\` is treated the same way). The tools are few and worth knowing well: detect (\`isna\`), drop (\`dropna\`) or fill (\`fillna\`, \`ffill\`).`, code: `import numpy as np
import pandas as pd

df = pd.DataFrame({
    "city":   ["Lyon", "Paris", None, "Nice", "Lyon"],
    "rooms":  [3, np.nan, 2, np.nan, 4],
    "price":  [320, 410, np.nan, np.nan, 390],
})
print(df, "\\n")
print("missing per column:", df.isna().sum().to_dict())
print("rows with no gaps:", len(df.dropna()), "of", len(df))
print("rows kept if at least 2 values are present:", len(df.dropna(thresh=2)))

filled = df.fillna({"city": "unknown", "rooms": df["rooms"].median(), "price": df["price"].mean()})
print("\\nfilled with a value per column:")
print(filled)
print("\\nforward fill (carry the last valid value down):")
print(df["price"].ffill().tolist())`, after: `- \`dropna()\` drops every row with **any** gap; \`how="all"\` drops only fully empty rows, \`thresh=k\` keeps rows with at least k values, and \`axis=1\` drops columns instead.
- \`fillna\` accepts one value, or a dictionary with a value per column. Filling with the mean or median is common, but compute that statistic on the **training data only** (see *Data leakage* below), ideally with \`SimpleImputer\` inside a pipeline.
- Forward fill (\`ffill\`) suits ordered data such as time series, where "the last known value" is a reasonable guess.
- A column of integers with a gap becomes floats, because NaN is a float. Use the nullable type \`"Int64"\` if you must keep integers.` },
      { type: 'code', title: "Duplicates, mapping and replacing values", code: `import numpy as np
import pandas as pd

food = pd.DataFrame({
    "item":   ["bacon", "pulled pork", "Bacon", "pastrami", "corned beef", "bacon", "nova lox"],
    "ounces": [4, 3, 12, 6, 7.5, 4, 6],
})
print("duplicate rows:", food.duplicated().tolist())
food = food.drop_duplicates()                         # keeps the first occurrence

animal = {"bacon": "pig", "pulled pork": "pig", "pastrami": "cow", "corned beef": "cow", "nova lox": "salmon"}
food["animal"] = food["item"].str.lower().map(animal)   # normalize case, then look up
print(food, "\\n")

readings = pd.Series([1.0, -999.0, 2.0, -999.0, -1000.0, 3.0])
print("sentinels replaced:", readings.replace([-999, -1000], np.nan).tolist())
print("renamed columns:", list(food.rename(columns=str.upper).columns))`, after: `- \`duplicated()\` flags rows identical to an earlier row; \`drop_duplicates(subset=["item"], keep="last")\` compares only some columns and keeps the last copy instead.
- \`map\` transforms each value with a dictionary or a function. Here "Bacon" had to be lower-cased first, or the lookup would miss it: **normalize text before matching**.
- \`replace\` swaps specific values, typically **sentinel codes** such as −999 that some system used to mean "missing". Left in place, a −999 silently drags every average down.
- \`rename\` relabels columns or rows with a dictionary or a function.` },
      { type: 'code', title: "Binning, outliers and random sampling", code: `import numpy as np
import pandas as pd

ages = pd.Series([20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32])
groups = pd.cut(ages, bins=[18, 25, 35, 60, 100], labels=["youth", "young adult", "middle aged", "senior"])
print("equal-meaning bins (pd.cut):")
print(groups.value_counts(sort=False).to_dict())
quartiles = pd.qcut(ages, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("equal-count bins (pd.qcut):", quartiles.value_counts(sort=False).to_dict())

rng = np.random.default_rng(12345)
data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["a", "b", "c"])
data.iloc[5, 1] = 7.5                                  # plant one outlier
outlier_rows = data[(data.abs() > 3).any(axis=1)]
print("\\nrows with any value beyond 3 standard deviations:", len(outlier_rows))
capped = data.clip(-3, 3)
print("largest value after capping at +/-3:", capped.values.max())

print("\\na random sample of 3 rows:")
print(data.sample(n=3, random_state=0).round(2))`, after: `- **\`pd.cut\`** splits by *value*: you choose the edges (by default each bin includes its right edge). **\`pd.qcut\`** splits by *quantiles*, so every bin holds about the same number of values. Both return a categorical with ordered labels.
- **Outliers**: with standardized data, |value| > 3 is a common first filter. It found 11 rows here although only one outlier was planted: in normally distributed data about 0.3% of values lie beyond 3 standard deviations by chance, so 3,000 values produce several. An extreme value is a reason to look, not proof of an error. Then decide case by case: fix data-entry errors, **cap** extreme values (\`clip\`), or keep them if they are real but rare.
- **Sampling**: \`sample(n)\` draws rows without replacement (set \`random_state\` to make it repeatable); \`replace=True\` draws with replacement, as in the bootstrap. \`np.random.permutation\` with \`take\` shuffles all rows.` },
      { type: 'code', title: "Strings and regular expressions", md: `Text columns need cleaning too: stray spaces, inconsistent case, information buried inside a longer string. Python's string methods handle simple cases; **regular expressions** (the \`re\` module) describe patterns.`, code: `import re
import pandas as pd

raw = "  apple, banana ,  cherry  "
pieces = [p.strip() for p in raw.split(",")]
print(pieces, "| joined:", " + ".join(pieces))

text = "Ana ana@example.com\\nBen ben.k@mail.example.org\\nChloe chloe@example.net"
pattern = re.compile(r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})", flags=re.IGNORECASE)
print("all addresses:", pattern.findall(text))           # one (user, domain, suffix) tuple each
print("hidden:", pattern.sub("[email]", text).splitlines())

emails = pd.Series({"Ana": "ana@example.com", "Ben": None, "Chloe": "chloe@example.net"})
print("\\nvectorized string methods handle missing values without errors:")
print(emails.str.contains("example.com").to_dict())
print(emails.str.split("@").str[0].to_dict())          # the part before the @`, after: `- \`split\`, \`strip\`, \`join\`, \`replace\`, \`lower\` and the \`in\` operator cover most simple cleaning.
- A regular expression such as \`[A-Z0-9._%+-]+@…\` matches a whole family of strings. \`findall\` returns every match, parentheses create **groups** that are returned separately, \`sub\` replaces matches, and \`re.IGNORECASE\` makes the letters case-insensitive. Compile a pattern once with \`re.compile\` when you use it repeatedly.
- On a pandas column, use the **\`.str\` accessor** (\`.str.contains\`, \`.str.split\`, \`.str.extract\`, \`.str[0:5]\`): it applies the method to every value and never raises an error on a missing one. A missing value comes back as missing (NaN), or, for true/false methods such as \`contains\` in recent pandas versions, as False; pass \`na=False\` to get the same result everywhere.` },
      { type: 'code', title: "Categorical data", md: `Columns with a few distinct values repeated many times (country, product type, size) can be stored as **categoricals**: an array of small integer **codes** plus one list of the distinct **categories**. They use less memory, make operations such as grouping faster, and can carry an order.`, code: `import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
fruit = pd.Series(rng.choice(["apple", "orange", "banana"], size=100_000))
cat = fruit.astype("category")
print("categories:", list(cat.cat.categories))
print("first codes:", cat.cat.codes[:6].tolist())
print(f"memory: text {fruit.memory_usage(deep=True) / 1e6:.1f} MB, categorical {cat.memory_usage(deep=True) / 1e6:.2f} MB")

sizes = pd.Categorical(["M", "S", "L", "M", "XL"], categories=["S", "M", "L", "XL"], ordered=True)
print("sorted by meaning, not alphabet:", list(sizes.sort_values()))
print("at least M:", list(sizes[sizes >= "M"]))

grades = pd.Series(["A", "B", "A"]).astype("category").cat.set_categories(["A", "B", "C"])
print("unused categories still count:", grades.value_counts(sort=False).to_dict())`, after: `- \`.cat.codes\` and \`.cat.categories\` expose the integer codes and the category list; \`pd.qcut\` and \`pd.cut\` return categoricals automatically.
- An **ordered** categorical sorts and compares by meaning (S < M < L < XL), not alphabetically.
- \`set_categories\`, \`add_categories\` and \`remove_unused_categories\` edit the category list; unused categories still appear in counts, which keeps tables consistent between datasets.
- For modelling, categories usually become one-hot columns (see [Feature Engineering](#/concept/feature-engineering)).` },
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
      { type: 'mcq', q: 'A model scores 99.8% accuracy on a problem experts find hard. What should you check first?', options: ['Whether to add more layers', 'Whether some feature leaks the label or test examples duplicate training ones', 'Whether to use a bigger test set', 'Nothing — ship it'], answer: 1, explain: 'Implausibly good results are usually leakage. Inspect the most important features and look for duplicates.' },
      {"type":"mcq","q":"A sensor writes −999 when a reading fails. What should you do before computing averages?","options":["Nothing: −999 is a valid number","Replace −999 with NaN (missing) and handle the gaps explicitly","Multiply the column by −1","Drop the whole column"],"answer":1,"explain":"A sentinel value is a missing value in disguise; left in place it distorts every statistic. `replace(-999, np.nan)` makes it a proper missing value."},
      {"type":"mcq","q":"What is the difference between `pd.cut` and `pd.qcut`?","options":["cut uses edges you choose (or equal-width bins); qcut uses quantiles so bins hold about equal counts","cut works on text, qcut on numbers","They are the same function","qcut removes outliers"],"answer":0,"explain":"cut bins by value; qcut bins by rank, so each bin contains roughly the same number of observations."},
      {"type":"tf","q":"pandas' vectorized string methods (`Series.str…`) raise an error when a value is missing.","answer":false,"explain":"They skip missing values and return NaN for them, which is one of their main advantages over looping with plain Python string methods."}
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
      { type: 'code', title: "Dummy variables with pandas", md: `\`pd.get_dummies\` turns a categorical column into **indicator** ("dummy") columns: one per category, 1 where the row has that category. Combined with binning, it can also turn a number into a set of range indicators.`, code: `import numpy as np
import pandas as pd

df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "value": range(6)})
print(pd.get_dummies(df["key"], prefix="key", dtype=int).join(df["value"]))

movies = pd.DataFrame({"title": ["Toy Story", "Heat", "Sabrina"],
                       "genres": ["Animation|Children's|Comedy", "Action|Crime|Thriller", "Comedy|Romance"]})
print("\\none row can belong to several genres:")
print(movies[["title"]].join(movies["genres"].str.get_dummies("|")))

values = np.array([0.04, 0.29, 0.62, 0.95, 0.51])
print("\\nbins as indicators:")
print(pd.get_dummies(pd.cut(values, [0, 0.2, 0.4, 0.6, 0.8, 1.0]), dtype=int))`, after: `- With \`drop_first=True\` one column per feature is dropped, since it is implied by the others ("not a and not c" means b). That matters for unregularized linear models; with regularization or trees it is unnecessary.
- \`str.get_dummies("|")\` handles **multi-valued** fields, where a row can have several categories at once.
- **Train/test consistency:** \`get_dummies\` creates columns from whatever values it sees, so a category that only appears in the test data creates an extra column, and a missing one drops a column. In a real model use \`OneHotEncoder(handle_unknown="ignore")\` inside a pipeline, which remembers the training categories.` },
      { type: 'text', title: 'Transforming and creating features', md: `- **Log transforms** for skewed, positive quantities (incomes, prices, counts): \`np.log1p(x)\`. They compress the long tail and often make relationships more linear.
- **Ratios and differences** that match the domain: price per m², debt-to-income, time since last purchase.
- **Date and time parts**: hour of day, day of week, month, "is weekend". Cyclical features can be encoded with sine/cosine so that 23:00 is close to 00:00.
- **Binning**: turning age into age groups when the effect is step-like.
- **Interactions and polynomials**: x₁·x₂ or x² let linear models fit curves — but add too many and you overfit (see *Overfitting*).
- **Text** becomes numbers via bag-of-words, TF-IDF or embeddings — the start of the NLP branch.

Always ask: *would this feature be available at prediction time?* — a feature computed from the future is leakage.` },
      { type: 'check', question: { type: 'mcq', q: 'Which model is essentially unaffected by standardizing its features?', options: ['k-nearest neighbours', 'A decision tree', 'Logistic regression with L2 regularization', 'A neural network'], answer: 1, explain: 'Trees choose thresholds on one feature at a time, so rescaling a feature just rescales its thresholds.' } },
      { type: 'text', title: "Binning: letting a linear model bend", md: `A linear model can only learn "y changes by w per unit of x". When the real relationship rises and falls, **binning** (discretisation) gives it more freedom: cut the range of x into intervals, one-hot encode the interval each value falls into, and the model learns a separate level for every interval.

\`\`\`python
from sklearn.preprocessing import KBinsDiscretizer
binner = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="uniform")
X_binned = binner.fit_transform(X_train)        # learn the bin edges on the training data only
\`\`\`

\`strategy="uniform"\` uses equal-width bins; \`strategy="quantile"\` puts the same number of training points into each bin, which copes better with skewed data. In pandas, \`pd.cut\` and \`pd.qcut\` do the same.

Binning helps linear models and makes features easy to explain ("customers aged 30 to 40"). It does little for tree-based models, which choose their own split points, and it throws away the order of the bins and the detail inside each one.` },
      { type: 'chart', title: "A linear model before and after binning its input", after: `On the raw feature a linear model can only draw one straight line through a wavy pattern. After binning x into 10 intervals and one-hot encoding them, the same linear model learns one level per bin: a step function that follows the wave. The price: the steps are crude, the model knows nothing about how neighbouring bins relate, and with too many bins each level rests on very few points.`, chart: {"kind":"line","series":[{"name":"data","x":[0.822,-1.381,-2.754,-2.901,1.88,2.477,0.64,1.377,0.262,2.61,1.895,-2.984,2.144,-2.798,1.378,-1.946,2.179,0.249,-1.202,-0.464,-2.83,-2.254,1.024,0.883,0.692,-0.698,2.983,2.885,1.113,0.903,1.131,-0.666,-2.189,1.329,0.152,-1.139,-0.085,2.337,2.604,-0.853,0.429,-1.069,0.566,-0.973,-0.65,2.342,-1.637,0.739,-2.496,1.996,1.723,-1.564,2.259,-2.649,-0.983,-2.098,-0.298,1.778,-1.616,-2.688,-0.573,-1.809,-2.455,0.482,-1.208,1.032,-1.803,2.653,-0.809,-2.367,0.775,2.563,-0.358,2.728,-0.001,-0.449,0.721,2.971,2.694,-0.24,1.546,-0.015,0.176,1.715,-0.512,1.407,1.267,2.592,-2.31,1.374,2.565,2.808,-2.912,2.182,2.887,2.743,-2.107,2.836,2.34,1.934],"y":[0.842,-1.205,0.024,-0.11,-0.065,-0.551,1.412,0.407,0.365,0.096,-0.711,-0.468,-0.443,-0.173,0.767,0.159,-0.076,0.324,-0.607,-0.722,-0.012,0.652,1.432,1.499,1.213,-1.622,0.543,0.144,0.7,1.321,0.94,-1.481,-0.025,0.944,0.453,-0.706,-0.199,0.014,0.323,-0.902,0.176,-0.796,1.177,-1.095,-1.047,-0.182,-0.263,1.11,-0.358,-0.185,-0.023,-0.159,-0.39,0.064,-1.473,0.087,-0.654,-0.315,-0.304,-0.05,-1.438,-0.804,0.398,0.877,-1.185,1.12,0.452,-0.048,-1.216,-0.156,1.726,0.128,-0.443,0.096,0.274,-0.805,1.392,0.51,-0.415,-0.225,-0.068,-0.108,0.336,-0.082,-0.824,0.684,0.82,0.043,0.16,1.212,-0.04,0.08,-1.013,-0.678,0.705,0.093,0.162,0.769,-0.682,-0.26],"scatter":true},{"name":"linear regression on x","x":[-3,-2.98,-2.96,-2.94,-2.92,-2.9,-2.88,-2.86,-2.84,-2.82,-2.8,-2.78,-2.76,-2.74,-2.72,-2.7,-2.68,-2.66,-2.64,-2.62,-2.6,-2.58,-2.56,-2.54,-2.52,-2.5,-2.48,-2.46,-2.44,-2.42,-2.4,-2.38,-2.36,-2.34,-2.32,-2.3,-2.28,-2.26,-2.24,-2.22,-2.2,-2.18,-2.16,-2.14,-2.12,-2.1,-2.08,-2.06,-2.04,-2.02,-2,-1.98,-1.96,-1.94,-1.92,-1.9,-1.88,-1.86,-1.84,-1.82,-1.8,-1.78,-1.76,-1.74,-1.72,-1.7,-1.68,-1.66,-1.64,-1.62,-1.6,-1.58,-1.56,-1.54,-1.52,-1.5,-1.48,-1.46,-1.44,-1.42,-1.4,-1.38,-1.36,-1.34,-1.32,-1.3,-1.28,-1.26,-1.24,-1.22,-1.2,-1.18,-1.16,-1.14,-1.12,-1.1,-1.08,-1.06,-1.04,-1.02,-1,-0.98,-0.96,-0.94,-0.92,-0.9,-0.88,-0.86,-0.84,-0.82,-0.8,-0.78,-0.76,-0.74,-0.72,-0.7,-0.68,-0.66,-0.64,-0.62,-0.6,-0.58,-0.56,-0.54,-0.52,-0.5,-0.48,-0.46,-0.44,-0.42,-0.4,-0.38,-0.36,-0.34,-0.32,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18,1.2,1.22,1.24,1.26,1.28,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.44,1.46,1.48,1.5,1.52,1.54,1.56,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.86,1.88,1.9,1.92,1.94,1.96,1.98,2,2.02,2.04,2.06,2.08,2.1,2.12,2.14,2.16,2.18,2.2,2.22,2.24,2.26,2.28,2.3,2.32,2.34,2.36,2.38,2.4,2.42,2.44,2.46,2.48,2.5,2.52,2.54,2.56,2.58,2.6,2.62,2.64,2.66,2.68,2.7,2.72,2.74,2.76,2.78,2.8,2.82,2.84,2.86,2.88,2.9,2.92,2.94,2.96,2.98,3],"y":[-0.368,-0.365,-0.363,-0.361,-0.359,-0.356,-0.354,-0.352,-0.349,-0.347,-0.345,-0.342,-0.34,-0.338,-0.335,-0.333,-0.331,-0.328,-0.326,-0.324,-0.321,-0.319,-0.317,-0.314,-0.312,-0.31,-0.308,-0.305,-0.303,-0.301,-0.298,-0.296,-0.294,-0.291,-0.289,-0.287,-0.284,-0.282,-0.28,-0.277,-0.275,-0.273,-0.27,-0.268,-0.266,-0.263,-0.261,-0.259,-0.256,-0.254,-0.252,-0.25,-0.247,-0.245,-0.243,-0.24,-0.238,-0.236,-0.233,-0.231,-0.229,-0.226,-0.224,-0.222,-0.219,-0.217,-0.215,-0.212,-0.21,-0.208,-0.205,-0.203,-0.201,-0.199,-0.196,-0.194,-0.192,-0.189,-0.187,-0.185,-0.182,-0.18,-0.178,-0.175,-0.173,-0.171,-0.168,-0.166,-0.164,-0.161,-0.159,-0.157,-0.154,-0.152,-0.15,-0.148,-0.145,-0.143,-0.141,-0.138,-0.136,-0.134,-0.131,-0.129,-0.127,-0.124,-0.122,-0.12,-0.117,-0.115,-0.113,-0.11,-0.108,-0.106,-0.103,-0.101,-0.099,-0.096,-0.094,-0.092,-0.09,-0.087,-0.085,-0.083,-0.08,-0.078,-0.076,-0.073,-0.071,-0.069,-0.066,-0.064,-0.062,-0.059,-0.057,-0.055,-0.052,-0.05,-0.048,-0.045,-0.043,-0.041,-0.039,-0.036,-0.034,-0.032,-0.029,-0.027,-0.025,-0.022,-0.02,-0.018,-0.015,-0.013,-0.011,-0.008,-0.006,-0.004,-0.001,0.001,0.003,0.006,0.008,0.01,0.012,0.015,0.017,0.019,0.022,0.024,0.026,0.029,0.031,0.033,0.036,0.038,0.04,0.043,0.045,0.047,0.05,0.052,0.054,0.057,0.059,0.061,0.064,0.066,0.068,0.07,0.073,0.075,0.077,0.08,0.082,0.084,0.087,0.089,0.091,0.094,0.096,0.098,0.101,0.103,0.105,0.108,0.11,0.112,0.115,0.117,0.119,0.121,0.124,0.126,0.128,0.131,0.133,0.135,0.138,0.14,0.142,0.145,0.147,0.149,0.152,0.154,0.156,0.159,0.161,0.163,0.166,0.168,0.17,0.172,0.175,0.177,0.179,0.182,0.184,0.186,0.189,0.191,0.193,0.196,0.198,0.2,0.203,0.205,0.207,0.21,0.212,0.214,0.217,0.219,0.221,0.224,0.226,0.228,0.23,0.233,0.235,0.237,0.24,0.242,0.244,0.247,0.249,0.251,0.254,0.256,0.258,0.261,0.263,0.265,0.268,0.27,0.272,0.275,0.277,0.279,0.281,0.284,0.286,0.288,0.291,0.293,0.295,0.298,0.3,0.302,0.305,0.307,0.309,0.312,0.314,0.316,0.319,0.321,0.323,0.326,0.328]},{"name":"linear regression on 10 one-hot bins","x":[-3,-2.98,-2.96,-2.94,-2.92,-2.9,-2.88,-2.86,-2.84,-2.82,-2.8,-2.78,-2.76,-2.74,-2.72,-2.7,-2.68,-2.66,-2.64,-2.62,-2.6,-2.58,-2.56,-2.54,-2.52,-2.5,-2.48,-2.46,-2.44,-2.42,-2.4,-2.38,-2.36,-2.34,-2.32,-2.3,-2.28,-2.26,-2.24,-2.22,-2.2,-2.18,-2.16,-2.14,-2.12,-2.1,-2.08,-2.06,-2.04,-2.02,-2,-1.98,-1.96,-1.94,-1.92,-1.9,-1.88,-1.86,-1.84,-1.82,-1.8,-1.78,-1.76,-1.74,-1.72,-1.7,-1.68,-1.66,-1.64,-1.62,-1.6,-1.58,-1.56,-1.54,-1.52,-1.5,-1.48,-1.46,-1.44,-1.42,-1.4,-1.38,-1.36,-1.34,-1.32,-1.3,-1.28,-1.26,-1.24,-1.22,-1.2,-1.18,-1.16,-1.14,-1.12,-1.1,-1.08,-1.06,-1.04,-1.02,-1,-0.98,-0.96,-0.94,-0.92,-0.9,-0.88,-0.86,-0.84,-0.82,-0.8,-0.78,-0.76,-0.74,-0.72,-0.7,-0.68,-0.66,-0.64,-0.62,-0.6,-0.58,-0.56,-0.54,-0.52,-0.5,-0.48,-0.46,-0.44,-0.42,-0.4,-0.38,-0.36,-0.34,-0.32,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18,1.2,1.22,1.24,1.26,1.28,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.44,1.46,1.48,1.5,1.52,1.54,1.56,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.86,1.88,1.9,1.92,1.94,1.96,1.98,2,2.02,2.04,2.06,2.08,2.1,2.12,2.14,2.16,2.18,2.2,2.22,2.24,2.26,2.28,2.3,2.32,2.34,2.36,2.38,2.4,2.42,2.44,2.46,2.48,2.5,2.52,2.54,2.56,2.58,2.6,2.62,2.64,2.66,2.68,2.7,2.72,2.74,2.76,2.78,2.8,2.82,2.84,2.86,2.88,2.9,2.92,2.94,2.96,2.98,3],"y":[-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,-0.17,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,0.076,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-0.621,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-1.149,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,-0.514,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,1.225,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,0.435,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,-0.332,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155,0.155]}],"xLabel":"input feature x","yLabel":"target y"} },
      { type: 'text', title: "Polynomial features and interactions", md: `Another way to let a linear model bend is to add **powers** of a feature (x², x³) and **interactions** between features (x₁·x₂). An interaction lets the effect of one feature depend on another: an extra bedroom is worth more in a large house than in a small one.

\`PolynomialFeatures(degree=k)\` generates every product of the inputs up to total degree k, and the count grows fast:

$$ number of features = C(d + k, k) − 1 $$
:::where
d — the number of input features
k — the polynomial degree
C(d + k, k) — "d + k choose k": the number of distinct products of the d inputs with total degree at most k
− 1 — leaving out the constant term, which the model's intercept already covers
:::

High powers also behave badly at the edges of the data (x⁵ explodes), and they need scaling and regularisation. Use a low degree, add only the interactions you have a reason to believe in (\`interaction_only=True\` keeps the products of different features and drops the pure powers), or let a model that finds interactions on its own, such as trees, kernel methods or neural networks, do the work.` },
      { type: 'chart', title: "How many polynomial features?", after: `With 10 inputs, degree 3 gives 285 features: manageable. With 100 inputs it gives 176,850, and with 1,000 inputs over 167 million. Full polynomial expansions only make sense for a handful of inputs; otherwise add the few interactions you have a reason to believe in, or use a model (trees, kernels, neural networks) that finds them for you.`, chart: {"kind":"line","x":[1,2,3,4,5],"yLog":true,"series":[{"name":"10 input features","y":[10,65,285,1000,3002]},{"name":"100 input features","y":[100,5150,176850,4598125,96560645]},{"name":"1000 input features","y":[1000,501500,167668500,42084793750,8459043543950]}],"xLabel":"polynomial degree","yLabel":"number of features (log scale)"} },
      { type: 'text', title: "Target encoding for high-cardinality categories", md: `A feature such as \`city\` with 400 values, or \`product_id\` with 100,000, makes one-hot encoding huge and sparse. **Target encoding** replaces each category by the average target of the training rows in that category ("the click rate of this city"), smoothed towards the overall average for small categories:

$$ enc(c) = ( n_c · ȳ_c + m · ȳ ) / ( n_c + m ) $$
:::where
c — a category, for example one city
n_c — the number of training rows in category c
ȳ_c — the average target within category c
ȳ — the overall average target
m — the smoothing strength: a category needs about m rows before its own average dominates
:::

It is powerful and **dangerously leaky**. Each row's own label is part of its category's average, so on the training data the encoding partly *is* the label, especially for small categories, and the model learns to trust it far too much. The fix is to compute the encoding **out of fold**: split the training data into k folds and encode each fold using statistics from the other folds only. scikit-learn's \`TargetEncoder\` does this automatically inside \`fit_transform\`.` },
      { type: 'code', title: "Naive versus out-of-fold target encoding", md: `The label here is pure noise: the city tells you nothing. Watch what each encoding claims anyway.`, code: `import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({"city": rng.integers(0, 400, n),          # 400 cities, about 5 rows each
                   "y": (rng.random(n) < 0.3).astype(int)})   # random labels
train, test = df.iloc[:1500].copy(), df.iloc[1500:].copy()
prior, m = train["y"].mean(), 10

stats = train.groupby("city")["y"].agg(["sum", "count"])
train["naive"] = train["city"].map(stats["sum"] / stats["count"])   # includes each row's own label

train["oof"] = np.nan
folds = np.arange(len(train)) % 5
for k in range(5):
    other = train[folds != k].groupby("city")["y"].agg(["sum", "count"])
    smooth = (other["sum"] + m * prior) / (other["count"] + m)
    train.loc[folds == k, "oof"] = train.loc[folds == k, "city"].map(smooth).fillna(prior).values

full = (stats["sum"] + m * prior) / (stats["count"] + m)
test["enc"] = test["city"].map(full).fillna(prior)

auc_naive = roc_auc_score(train["y"], train["naive"])
print(f"training rows, naive encoding:        AUC {auc_naive:.2f}")
print(f"training rows, out-of-fold encoding:  AUC {roc_auc_score(train['y'], train['oof']):.2f}")
print(f"test rows:                            AUC {roc_auc_score(test['y'], test['enc']):.2f}")
print("naive encoding leaks:", auc_naive > 0.7)`, after: `On the training rows the naive encoding looks strongly predictive (AUC around 0.79) even though the labels are random: with about 4 rows per city, each row's own label moves its city's average a lot. A model trained on it would lean on a feature that is worthless on new data, where the AUC is close to 0.5. The out-of-fold version tells the truth on the training rows too.`, predict: {"type":"mcq","q":"The labels are random. How predictive will the naive target encoding look on the training rows?","options":["Not at all: AUC about 0.5","Clearly predictive: AUC well above 0.5","Perfect: AUC exactly 1.0","Worse than random: AUC well below 0.5"],"answer":1,"explain":"Each row's own label is included in its category's average, so on the training data the encoding partly copies the label. The effect is strongest for small categories.","expect":["naive encoding leaks: True"]} },
      { type: 'text', title: "Choosing features: filter, wrapper and embedded methods", md: `More features are not always better: irrelevant ones add noise, cost memory and time, and make models harder to explain. Three families of methods pick a subset:

| Family | How it works | Examples in scikit-learn | Trade-off |
| --- | --- | --- | --- |
| Filter | score each feature on its own against the target and keep the best | \`SelectKBest\` with \`f_classif\` or \`mutual_info_classif\`; \`VarianceThreshold\` | fast, but blind to interactions between features |
| Wrapper | train models on different subsets and keep the best subset | \`RFE\` (recursive feature elimination), \`SequentialFeatureSelector\` | accounts for interactions, but expensive |
| Embedded | the model selects while it trains | L1-regularised (lasso) models; tree importances with \`SelectFromModel\` | a good middle ground |

Whatever you use, **select inside the cross-validation loop**, on the training folds only, by putting the selector in a [pipeline](#/concept/sklearn-workflow). Selecting features on the full dataset and then cross-validating is a famous leak: among enough random features, some correlate with the target by chance, and the cross-validated score looks excellent on pure noise.` }
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
      { type: 'multi', q: 'Which models usually benefit from feature scaling? (select all)', options: ['k-nearest neighbours', 'Random forest', 'Neural network', 'Support vector machine'], answer: [0, 2, 3], explain: 'Distance and gradient-based models are sensitive to scale; tree ensembles are not.' },
      {"type":"mcq","q":"With 20 input features, how many features does PolynomialFeatures(degree=2) produce, leaving out the constant term?","options":["40","210","230","400"],"answer":2,"explain":"C(20 + 2, 2) − 1 = 231 − 1 = 230: the 20 originals, 20 squares and 190 pairwise products."},
      {"type":"mcq","q":"Why should target encoding be computed out of fold on the training data?","options":["It is faster","Otherwise each row's own label leaks into its encoding and the feature looks more predictive than it is","Out-of-fold encodings need no smoothing","Because the test data has no categories"],"answer":1,"explain":"Including a row's own label in its category average makes the encoding partly a copy of the label on the training data."},
      {"type":"tf","q":"Selecting the best features on the full dataset and then cross-validating gives an honest estimate of performance.","answer":false,"explain":"The selection already looked at the validation folds' labels. Put the selector inside a pipeline so that it is refitted on the training folds only."}
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
