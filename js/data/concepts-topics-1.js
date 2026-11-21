/* ML Atlas – content: graphical models and HMMs, GLMs and survival analysis, 3D vision, AI for science, audio generation */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'graphical-models', branch: 'ml', title: 'Probabilistic Graphical Models & Hidden Markov Models', short: 'Graphical Models & HMMs', level: 3, minutes: 50, prereqs: ['bayesian-methods', 'naive-bayes'],
    tags: ['graphical models', 'bayesian network', 'markov random field', 'conditional independence', 'explaining away', 'hidden markov model', 'hmm', 'forward algorithm', 'viterbi', 'baum-welch', 'crf', 'kalman filter', 'probabilistic programming'],
    summary: 'Describing joint probability distributions with graphs: Bayesian networks and Markov random fields, conditional independence and explaining away, exact inference by enumeration, and hidden Markov models with the forward algorithm (the probability of a sequence) and the Viterbi algorithm (the most likely hidden path); learning with Baum–Welch, and where graphical models live on inside modern deep learning.',
    why: `A joint distribution over many variables is impossibly large to write down: 30 yes-or-no variables already need about a billion probabilities. Graphical models make such distributions manageable by saying which variables directly influence which, and the structure pays off twice: far fewer numbers to learn, and efficient algorithms that reuse partial results. Hidden Markov models, the most famous example, powered speech recognition for decades and still underlie sequence tagging, gene finding and many time-series methods.`,
    objectives: [
      'Read a Bayesian network as a factorization of a joint distribution, and count the parameters it saves',
      'Compute conditional probabilities by enumeration, and explain the "explaining away" effect',
      'Describe a hidden Markov model and compute sequence probabilities with the forward algorithm',
      'Find the most likely hidden state sequence with the Viterbi algorithm'
    ],
    sections: [
      { type: 'text', title: "Why describe probabilities with graphs?", md: `A joint distribution over many variables quickly becomes too large to write down. Thirty yes-or-no variables have 2³⁰ combinations, so a full probability table needs about a billion numbers, far more than any dataset could estimate.

Most variables, though, influence each other only through a few others. A **Bayesian network** records this in a directed acyclic graph: each variable has a small table of probabilities given its *parents*, and the joint distribution is their product:

$$ P(x₁, …, x_n) = Π_i P(x_i | parents(x_i)) $$
:::where
x₁, …, x_n — the variables, such as rain, sprinkler and wet grass
parents(x_i) — the variables with an arrow pointing into x_i in the graph
P(x_i | parents(x_i)) — a small table: the distribution of x_i for each combination of its parents' values
Π_i — the product over all the variables
:::

With binary variables and at most k parents each, the network needs about n · 2ᵏ numbers instead of 2ⁿ: for 30 variables with three parents each, 240 instead of a billion. Naive Bayes is the simplest example: the class is the single parent of every feature (see [Naive Bayes](#/concept/naive-bayes)).` },
      { type: 'text', title: "Independence, explaining away and Markov random fields", md: `Three small patterns explain how information flows through a Bayesian network:

- **Chain** A → B → C: knowing B makes A irrelevant for C. If smoking causes tar deposits and tar causes cancer, then once you know the tar level, smoking adds nothing.
- **Fork** A ← B → C: a common cause makes A and C correlated until you condition on it. Ice-cream sales and sunburn both follow the weather.
- **Collider** A → C ← B: two independent causes of one effect stay independent, until the effect is observed. Then they become dependent: if the grass is wet and you learn the sprinkler was on, rain becomes less likely. This is **explaining away**.

**Markov random fields** use undirected edges instead: they score each group of connected variables with a non-negative *potential* and normalize, which suits symmetric relationships such as neighbouring pixels in an image. Their conditional cousins, **conditional random fields** (CRFs), label sequences such as the words of a sentence.

**Inference**, computing probabilities given evidence, can be exact (enumeration for tiny models, variable elimination, belief propagation on tree-shaped graphs) or approximate (sampling with MCMC, or variational inference; see [Bayesian Methods](#/concept/bayesian-methods)).` },
      { type: 'code', title: "Explaining away in the sprinkler network", md: `The classic network: rain (R) affects whether the sprinkler (S) is used, and both can wet the grass (W). Every query is answered by adding up the probabilities of all eight possible worlds:`, code: `import itertools
P_rain = 0.2
P_sprinkler = {True: 0.01, False: 0.4}                 # P(sprinkler on | rain), P(sprinkler on | no rain)
P_wet = {(True, True): 0.99, (True, False): 0.8, (False, True): 0.9, (False, False): 0.0}   # P(wet | rain, sprinkler)

def joint(rain, sprinkler, wet):                       # the network's factorization: P(r) P(s | r) P(w | r, s)
    p = (P_rain if rain else 1 - P_rain) * (P_sprinkler[rain] if sprinkler else 1 - P_sprinkler[rain])
    pw = P_wet[(rain, sprinkler)]
    return p * (pw if wet else 1 - pw)

def prob(query, evidence):                             # P(query | evidence), by adding up all 8 possible worlds
    num = den = 0.0
    for r, s, w in itertools.product([True, False], repeat=3):
        world = {"rain": r, "sprinkler": s, "wet": w}
        if all(world[k] == v for k, v in evidence.items()):
            p = joint(r, s, w)
            den += p
            if all(world[k] == v for k, v in query.items()):
                num += p
    return num / den

print(f"P(rain)                           = {prob({'rain': True}, {}):.3f}")
print(f"P(rain | grass wet)               = {prob({'rain': True}, {'wet': True}):.3f}")
print(f"P(rain | grass wet, sprinkler on) = {prob({'rain': True}, {'wet': True, 'sprinkler': True}):.3f}")
print("the sprinkler explains the wet grass away:", prob({'rain': True}, {'wet': True, 'sprinkler': True}) < prob({'rain': True}, {'wet': True}))`, after: `Seeing wet grass raises the probability of rain from 0.20 to 0.36. Learning that the sprinkler was on then drops it to under 0.01: the sprinkler accounts for the wet grass, so rain is no longer needed to explain it (and in this garden the sprinkler is rarely used when it rains). Enumeration is exact but visits every combination of the variables, which is only possible for tiny networks; real systems use smarter algorithms that exploit the graph.`, predict: {"type":"mcq","q":"The grass is wet, which makes rain more likely (about 0.36). Then you learn that the sprinkler was on. What happens to the probability of rain?","options":["It rises further","It stays the same","It drops sharply","It becomes exactly zero"],"answer":2,"explain":"It drops to about 0.007. The sprinkler explains the wet grass, so rain is no longer needed: two causes of the same effect compete once the effect is observed.","expect":["the sprinkler explains the wet grass away: True"]} },
      { type: 'check', question: {"type":"mcq","q":"Two independent causes A and B can each produce an effect C (A → C ← B). When do A and B become dependent?","options":["Never","Always","When C is observed","When A is observed"],"answer":2,"explain":"Observing a common effect couples its causes: if C happened and A did not, B probably did. This is explaining away."} },
      { type: 'text', title: "Hidden Markov models", md: `A **hidden Markov model** (HMM) describes a sequence of observations produced by a sequence of states that you cannot see. The hidden state follows a Markov chain (tomorrow depends only on today), and each observation depends only on the current state:

$$ P(x_{1:T}, z_{1:T}) = π(z₁) · B(z₁, x₁) · Π_{t=2..T} A(z_{t−1}, z_t) · B(z_t, x_t) $$
:::where
z_t — the hidden state at time t, one of K possible states
x_t — the observation at time t
π(z) — the probability of starting in state z
A(i, j) — the transition probability of moving from state i to state j
B(z, x) — the emission probability of observing x in state z
T — the length of the sequence
:::

Three questions come up again and again:

1. **Likelihood:** how probable is an observed sequence? (the forward algorithm)
2. **Decoding:** which hidden sequence most likely produced it? (the Viterbi algorithm)
3. **Learning:** what are π, A and B, given only observations? (the Baum–Welch algorithm, an EM algorithm like the one in [Gaussian Mixtures, EM & Other Clustering](#/concept/gmm-em))

HMMs ran speech recognition for decades before deep learning, and are still used for part-of-speech tagging, gene finding, activity recognition from sensors and detecting regimes in financial time series.` },
      { type: 'code', title: "The forward algorithm and the Viterbi algorithm", md: `A classic teaching example: the weather on each of ten days (hot or cold) is hidden, but we know how many ice creams someone ate (1 to 3). Summing over every possible weather sequence directly, and with the forward algorithm:`, code: `import itertools
import numpy as np
states = ["hot", "cold"]
pi = np.array([0.8, 0.2])                              # weather on the first day
A = np.array([[0.7, 0.3], [0.4, 0.6]])                 # A[i, j] = P(tomorrow is j | today is i)
B = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])       # B[i, k] = P(k + 1 ice creams eaten | weather i)
obs = [2, 0, 2, 1, 0, 0, 1, 2, 2, 1]                   # ice creams eaten on ten days, minus one

def forward(obs):                                      # P(observations), summing over hidden paths step by step
    alpha = pi * B[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
    return alpha.sum()

def brute_force(obs):                                  # the same sum, one hidden path at a time
    total = 0.0
    for path in itertools.product(range(2), repeat=len(obs)):
        p = pi[path[0]] * B[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
        total += p
    return total

def viterbi(obs):                                      # the single most likely hidden path
    delta, back = np.log(pi) + np.log(B[:, obs[0]]), []
    for o in obs[1:]:
        scores = delta[:, None] + np.log(A)            # scores[i, j]: best path ending in i, then a move to j
        back.append(scores.argmax(axis=0))
        delta = scores.max(axis=0) + np.log(B[:, o])
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.insert(0, int(b[path[0]]))
    return [states[s] for s in path]

print(f"P(observations), forward algorithm:        {forward(obs):.6e}")
print(f"P(observations), summing all {2 ** len(obs)} paths:   {brute_force(obs):.6e}")
print("most likely weather:", " ".join(viterbi(obs)))
print("the forward algorithm gives the same answer:", bool(np.isclose(forward(obs), brute_force(obs))))`, after: `Both methods give the same probability, but the brute-force sum visits all 1,024 weather sequences, while the forward algorithm keeps one number per state and updates it once per day: 10 steps of 2 × 2 work. The trick is that every path through "hot on day 5" shares the same past, so the past can be summed once. The Viterbi algorithm uses the same table with a maximum instead of a sum, plus pointers back to the best previous state, and works with logarithms so that products of many small probabilities do not underflow to zero.`, predict: {"type":"mcq","q":"Ten days, with two possible hidden states on each day. How many hidden weather sequences does the brute-force sum go through?","options":["20","100","1,024","3,628,800"],"answer":2,"explain":"2 choices on each of 10 days give 2¹⁰ = 1,024 sequences. The forward algorithm avoids listing them, so its cost grows only linearly with the number of days.","expect":["the forward algorithm gives the same answer: True"]} },
      { type: 'chart', title: "Summing over hidden paths: brute force against the forward algorithm", md: `Multiplications needed to compute P(observations) for a hidden Markov model as the sequence gets longer: about 2T for each of the Kᵀ paths when summing them one by one, against K² + K per step for the forward algorithm:`, after: `The brute-force sum doubles with every extra day, so 40 days of two-state weather need about 10¹⁴ multiplications. The forward algorithm grows in a straight line: even with 10 hidden states, 40 days take 4,400. Paths that share a past share its sum, and dynamic programming computes that sum once.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],"yLog":true,"points":false,"series":[{"name":"every path, 2 states","y":[4,16,48,128,320,768,1792,4096,9216,20480,45056,98304,212992,458752,983040,2097152,4456448,9437184,19922944,41943040,88080384,184549376,385875968,805306368,1677721600,3489660928,7247757312,15032385536,31138512896,64424509440,133143986176,274877906944,566935683072,1168231104512,2405181685760,4947802324992,10170482556928,20890720927744,42880953483264,87960930222080]},{"name":"forward algorithm, 2 states","y":[6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240]},{"name":"forward algorithm, 10 states","y":[110,220,330,440,550,660,770,880,990,1100,1210,1320,1430,1540,1650,1760,1870,1980,2090,2200,2310,2420,2530,2640,2750,2860,2970,3080,3190,3300,3410,3520,3630,3740,3850,3960,4070,4180,4290,4400]}],"xLabel":"length of the observed sequence (days)","yLabel":"multiplications (log scale)"} },
      { type: 'text', title: "Learning, and where graphical models live today", md: `- **Learning the numbers.** With every variable observed, the tables are simple counts (plus smoothing). With hidden variables, EM alternates between inferring the hidden values and re-estimating the tables: Baum–Welch is EM for HMMs.
- **Learning the structure.** Finding the graph itself from data is hard, because the number of possible graphs explodes, so structure usually comes from domain knowledge, with data used to check it (see [Causal Inference](#/concept/causal-inference) for when arrows mean causes).
- **Inside modern deep learning**, the ideas are everywhere: CRF layers on top of neural sequence taggers, Kalman filters (a continuous, linear-Gaussian HMM) for tracking, variational autoencoders and diffusion models as latent-variable models, and state-space models for long sequences (see [State-Space Models](#/concept/ssm-architectures)).
- **Probabilistic programming** languages such as PyMC, Stan and NumPyro let you write a model as code and leave the inference to the library.` }
    ],
    when: `- You know (or can guess) the structure of the problem: which causes affect which effects.
- Sequences with hidden states: tagging words, segmenting genomes, detecting regimes in time series.
- You need calibrated probabilities and answers to "what if we observe X?" questions, with little data.`,
    pitfalls: `- Writing down full joint tables: they grow exponentially. Factorize.
- Treating a Bayesian network's arrows as proven causes: they encode conditional independences, which many causal stories share.
- Multiplying many small probabilities directly: use logarithms, as the Viterbi code does, or the numbers underflow to zero.
- Forgetting that HMMs assume the next state depends only on the current one (the Markov assumption).`,
    takeaways: [
      'A Bayesian network factorizes a joint distribution into one small table per variable, conditioned on its parents.',
      'Observing a common effect makes its causes dependent: one cause "explains away" the other.',
      'Hidden Markov models pair hidden states that follow a Markov chain with observations that depend on the current state.',
      'The forward algorithm sums over all hidden paths in T·K² steps instead of K^T, and Viterbi finds the single best path the same way.',
      'Graphical-model ideas live on in CRF layers, variational autoencoders, diffusion models and state-space models.'
    ],
    quiz: [
      {"type":"fill","q":"How many free parameters does a full joint probability table over 10 binary variables need? (2¹⁰ − 1)","answer":["1023","1,023"],"explain":"2¹⁰ = 1,024 combinations, minus one because the probabilities must sum to 1."},
      {"type":"mcq","q":"In a chain A → B → C, what does observing B do?","options":["Makes A and C dependent","Makes A and C independent","Changes the graph","Nothing"],"answer":1,"explain":"B screens A off from C: once B is known, A carries no further information about C."},
      {"type":"mcq","q":"What does the Viterbi algorithm return?","options":["The probability of the observations","The single most likely sequence of hidden states","The transition matrix","A random sample of hidden states"],"answer":1,"explain":"Viterbi replaces the forward algorithm's sums by maxima and keeps back-pointers to recover the best path."},
      {"type":"tf","q":"The cost of the forward algorithm grows exponentially with the length of the sequence.","answer":false,"explain":"It grows linearly with the length, as T · K² for K states, because it sums over paths step by step instead of listing them."},
      {"type":"mcq","q":"What is the Baum–Welch algorithm?","options":["A sampling method for Bayesian networks","An EM algorithm that learns an HMM's probabilities from observations alone","A way to draw graphs","A test for independence"],"answer":1,"explain":"It alternates between inferring the hidden states (with forward-backward) and re-estimating π, A and B."},
      {"type":"mcq","q":"Why does the Viterbi code work with logarithms of probabilities?","options":["Logarithms are faster to compute","Multiplying many small probabilities underflows to zero, while adding logarithms does not","It changes the answer","The probabilities are negative"],"answer":1,"explain":"A product of hundreds of probabilities can be smaller than the smallest representable number; sums of logarithms stay in range and preserve the order."}
    ],
    problems: [
      {
        id: 'hmm-forward', title: 'The Forward Algorithm', difficulty: 'medium',
        prompt: `A hidden Markov model has an initial distribution \`pi\` (a list of K probabilities), a transition matrix \`A\` (A[i][j] = probability of moving from state i to state j) and an emission matrix \`B\` (B[i][k] = probability of observing symbol k in state i). Implement \`forward(pi, A, B, obs)\` that returns the probability of the observation sequence \`obs\` (a list of symbol indices), summed over all hidden paths.

Use the forward recursion: start with α = pi × B[:, obs[0]] (elementwise), then for every next observation o, α ← (α · A) × B[:, o]. The answer is the sum of the final α.

\`\`\`text
forward([0.5, 0.5], [[1, 0], [0, 1]], [[0.9, 0.1], [0.2, 0.8]], [0])   ->  0.55
\`\`\``,
        starter: `
import numpy as np

def forward(pi, A, B, obs):
    # TODO: alpha = pi * B[:, obs[0]], then alpha = (alpha @ A) * B[:, o] for each next observation
    pass
`,
        solution: `
import numpy as np

def forward(pi, A, B, obs):
    pi, A, B = np.asarray(pi, float), np.asarray(A, float), np.asarray(B, float)
    alpha = pi * B[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
    return float(alpha.sum())
`,
        hints: ['Convert the inputs to NumPy arrays first.', 'B[:, o] is the column of emission probabilities for symbol o.', 'alpha @ A sums over the previous state for every next state.'],
        explain: `Each α value is the probability of the observations so far, ending in a given state. Updating it step by step reuses the sums over all earlier paths, so the cost grows with T · K² instead of K^T. The same idea, with maximum instead of sum, gives the Viterbi algorithm.`,
        tests: [
          { call: 'round(forward([0.5, 0.5], [[1, 0], [0, 1]], [[0.9, 0.1], [0.2, 0.8]], [0]), 6)', expect: '0.55' },
          { name: 'two steps', call: 'round(forward([1.0, 0.0], [[0.5, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [0, 1]), 6)', expect: '0.5' },
          { name: 'the ice-cream example', call: 'abs(forward([0.8, 0.2], [[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]], [2, 0, 2, 1, 0, 0, 1, 2, 2, 1]) - 1.458069e-05) < 1e-10', expect: 'True' },
          { name: 'a random model vs reference', setup: 'import numpy as np\nrng = np.random.default_rng(3)\nA_ = rng.random((3, 3)); A_ /= A_.sum(1, keepdims=True)\nB_ = rng.random((3, 4)); B_ /= B_.sum(1, keepdims=True)\npi_ = np.ones(3) / 3\nobs_ = list(rng.integers(0, 4, 12))', call: 'round(forward(pi_, A_, B_, obs_) * 1e9, 6)', hidden: true }
        ]
      },
      {
        id: 'hmm-viterbi', title: 'The Viterbi Algorithm', difficulty: 'medium',
        prompt: `Implement \`viterbi(pi, A, B, obs)\` for the same hidden Markov model as in the forward algorithm: return the **most likely sequence of hidden states** (a list of state indices) for the observations \`obs\`.

Work with logarithms to avoid underflow:

- δ = log pi + log B[:, obs[0]];
- for every next observation o: scores[i][j] = δ[i] + log A[i][j]; remember, for each j, the best previous state i; then δ[j] = max over i of scores[i][j] + log B[j][o];
- the last state is the argmax of δ; follow the remembered best previous states backwards.

You may assume that no probability on the best path is zero (\`np.log(0)\` is \`-inf\`, which still works with max).`,
        starter: `
import numpy as np

def viterbi(pi, A, B, obs):
    # TODO: log-space dynamic programming with back-pointers
    pass
`,
        solution: `
import numpy as np

def viterbi(pi, A, B, obs):
    with np.errstate(divide='ignore'):
        lpi, lA, lB = np.log(np.asarray(pi, float)), np.log(np.asarray(A, float)), np.log(np.asarray(B, float))
    delta, back = lpi + lB[:, obs[0]], []
    for o in obs[1:]:
        scores = delta[:, None] + lA
        back.append(scores.argmax(axis=0))
        delta = scores.max(axis=0) + lB[:, o]
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.insert(0, int(b[path[0]]))
    return path
`,
        hints: ['scores = delta[:, None] + log A has one row per previous state and one column per next state.', 'Store scores.argmax(axis=0) at every step: the best previous state for each next state.', 'Rebuild the path from the end: path.insert(0, back_t[path[0]]).'],
        explain: `Viterbi is the forward algorithm with the sum replaced by a maximum, plus back-pointers. It finds the single most probable hidden path, which can differ from choosing the most probable state at each time separately: those states may not even form a possible path.`,
        tests: [
          { call: 'viterbi([0.8, 0.2], [[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]], [2, 0, 2, 1, 0, 0, 1, 2, 2, 1])', expect: '[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]' },
          { name: 'sticky states', call: 'viterbi([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[0.8, 0.2], [0.2, 0.8]], [0, 0, 1, 1, 1])', expect: '[0, 0, 1, 1, 1]' },
          { name: 'a single observation', call: 'viterbi([0.3, 0.7], [[0.5, 0.5], [0.5, 0.5]], [[0.9, 0.1], [0.5, 0.5]], [0])', expect: '[1]' },
          { name: 'a random model vs reference', setup: 'import numpy as np\nrng = np.random.default_rng(5)\nA_ = rng.random((3, 3)); A_ /= A_.sum(1, keepdims=True)\nB_ = rng.random((3, 4)); B_ /= B_.sum(1, keepdims=True)\npi_ = np.ones(3) / 3\nobs_ = list(rng.integers(0, 4, 15))', call: 'viterbi(pi_, A_, B_, obs_)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Stanford CS228, Probabilistic Graphical Models (course notes)', url: 'https://ermongroup.github.io/cs228-notes/' },
      { title: 'Jurafsky & Martin, Speech and Language Processing (hidden Markov models and tagging)', url: 'https://web.stanford.edu/~jurafsky/slp3/' }
    ]
  },

  // =====================================================================================
  {
    id: 'glm-survival', branch: 'ml', title: 'Generalized Linear Models & Survival Analysis', short: 'GLMs & Survival', level: 2, minutes: 50, prereqs: ['logistic-regression', 'mle-map'],
    tags: ['generalized linear model', 'glm', 'link function', 'exponential family', 'poisson regression', 'gamma regression', 'deviance', 'overdispersion', 'offset', 'survival analysis', 'censoring', 'kaplan-meier', 'hazard', 'cox model', 'concordance index'],
    summary: 'One framework for targets that are not normally distributed: generalized linear models with link functions (Poisson regression for counts, logistic for yes/no, gamma for positive amounts), exposure offsets and overdispersion; then time-to-event data, censoring, the Kaplan–Meier estimator, hazards, the Cox proportional-hazards model and the concordance index.',
    why: `Many quantities that matter are not bell-shaped numbers: the number of support tickets, insurance claims or website visits (counts), the size of a claim (positive and skewed), the time until a customer cancels or a machine fails. Ordinary regression mishandles them, predicting negative counts or ignoring customers who have not cancelled yet. Generalized linear models and survival analysis are the standard, interpretable tools for these targets in insurance, medicine, reliability and churn.`,
    objectives: [
      'Choose a distribution and link function for counts, binary outcomes and positive amounts',
      'Fit and interpret a Poisson regression, with coefficients as rate ratios',
      'Explain censoring and estimate a survival curve with the Kaplan–Meier method',
      'Describe hazards, the Cox model and the concordance index'
    ],
    sections: [
      { type: 'text', title: "One framework for many kinds of targets", md: `Linear regression assumes the target is a number with normally distributed noise. Many targets are not like that: counts (0, 1, 2, …), yes/no outcomes, positive amounts with a long right tail. **Generalized linear models** (GLMs) keep the linear predictor but connect it to the target's mean through a *link function*, and use a distribution that suits the target:

$$ g( E[y | x] ) = β₀ + β₁ x₁ + … + β_p x_p $$
:::where
y — the target, such as a count of support tickets
x₁, …, x_p — the features
E[y | x] — the expected value of the target for these features
g — the link function: identity, logit or log, depending on the target
β₀, …, β_p — the coefficients, fitted by maximum likelihood
:::

| Target | Distribution | Link | Example |
| --- | --- | --- | --- |
| any real number | normal | identity | temperature |
| yes or no | Bernoulli | logit | will this customer churn? (logistic regression) |
| count | Poisson, or negative binomial | log | tickets per month |
| positive and skewed | gamma | log | size of an insurance claim |

All of them are fitted by maximum likelihood (see [MLE & MAP](#/concept/mle-map)), and their natural error measure is the **deviance**, which plays the role that squared error plays for linear regression.` },
      { type: 'code', title: "Poisson regression for counts", md: `Support tickets per month for 2,000 customer accounts, which depend on how many people use the account and on whether it still runs the old version of the product:`, code: `import numpy as np, warnings
from sklearn.linear_model import PoissonRegressor, LinearRegression
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 2000
users = rng.uniform(0, 10, n)                          # thousands of active users of each customer account
legacy = (rng.random(n) < 0.3).astype(float)           # 1 if the account still runs the old version
X = np.column_stack([users, legacy])
rate = np.exp(-0.5 + 0.25 * users + 0.6 * legacy)      # support tickets per month follow a log-linear model
y = rng.poisson(rate)
Xtr, Xte, ytr, yte = X[:1500], X[1500:], y[:1500], y[1500:]

lin = LinearRegression().fit(Xtr, ytr)
poi = PoissonRegressor(alpha=0).fit(Xtr, ytr)
def deviance(y, mu):                                   # mean Poisson deviance: lower means better predicted counts
    mu = np.clip(mu, 1e-9, None)
    return 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1) / mu), 0) - (y - mu))
print("Poisson coefficients:", poi.coef_.round(3), "  (true: 0.25 and 0.6)")
print(f"legacy accounts file exp({poi.coef_[1]:.2f}) = {np.exp(poi.coef_[1]):.2f} times as many tickets")
pl, pp = lin.predict(Xte), poi.predict(Xte)
print(f"linear regression:  lowest prediction {pl.min():+.2f}, test deviance {deviance(yte, pl):.3f}")
print(f"Poisson regression: lowest prediction {pp.min():+.2f}, test deviance {deviance(yte, pp):.3f}")
print("legacy accounts file about 1.8 times as many tickets:", bool(1.6 < np.exp(poi.coef_[1]) < 2.0))
print("linear regression predicts negative counts:", bool(pl.min() < 0))`, after: `The Poisson model recovers the true coefficients almost exactly. Because of the log link, they act multiplicatively: exp(0.6) ≈ 1.8, so legacy accounts file about 1.8 times as many tickets, and each extra thousand users multiplies the rate by exp(0.25) ≈ 1.3. Linear regression, which assumes effects add up, predicts negative ticket counts for small accounts and has five times the deviance on new data.`, predict: {"type":"mcq","q":"In a Poisson regression with a log link, the coefficient of \"legacy version\" is 0.6. What does that mean?","options":["Legacy accounts file 0.6 more tickets per month","Legacy accounts file about 1.8 times as many tickets","60% of tickets come from legacy accounts","The model is 60% accurate"],"answer":1,"explain":"With a log link, coefficients multiply the expected count: exp(0.6) ≈ 1.82, a rate ratio.","expect":["legacy accounts file about 1.8 times as many tickets: True"]} },
      { type: 'text', title: "Exposure, overdispersion and many zeros", md: `- **Exposure.** Counts depend on how long or how much each unit was observed: claims per policy-year, visits per day. Add log(exposure) to the linear predictor as an *offset* (a term with coefficient fixed at 1), so the model learns rates rather than raw counts.
- **Overdispersion.** A Poisson distribution has variance equal to its mean, but real counts usually vary more, because of unobserved differences between units. The fitted means can still be fine, but standard errors and intervals come out too narrow. Negative binomial regression adds a parameter for the extra variance.
- **Excess zeros.** Many customers never file a ticket at all. Zero-inflated models treat "never" as a separate process, and Tweedie models (used heavily in insurance pricing) handle targets that are often exactly zero and otherwise positive and continuous.` },
      { type: 'text', title: "Survival analysis: time until an event", md: `Many questions ask *how long until* something happens: a customer cancels, a machine fails, a patient relapses, a user converts. The difficulty is **censoring**: when you analyse the data, many subjects have not had the event yet. A customer who joined four months ago and is still subscribed tells you their time to cancel is *more than* four months, not what it is.

Survival analysis describes such times with two functions:

$$ S(t) = P(T > t)          h(t) = lim_{Δt→0} P(t ≤ T < t + Δt | T ≥ t) / Δt $$
:::where
T — the time until the event, a random variable
S(t) — the survival function: the probability that the event has not happened by time t
h(t) — the hazard: the instantaneous rate of the event at time t, among those for whom it has not happened yet
Δt — a small interval of time
:::

The **Kaplan–Meier estimator** builds S(t) from data, using every subject for as long as they were observed: at each event time, survival is multiplied by (1 − events / subjects still at risk), and censored subjects simply leave the risk set when their observation ends.` },
      { type: 'code', title: "Kaplan–Meier, and what goes wrong without it", md: `Five hundred customers with a true median time to cancel of 8.3 months, each watched for a random period of up to 18 months:`, code: `import numpy as np
rng = np.random.default_rng(1)
n = 500
true_time = rng.exponential(12.0, n)                   # months until each customer cancels (true median 12 ln 2 = 8.3)
followup = rng.uniform(0, 18, n)                       # how long each customer has been watched so far
time = np.minimum(true_time, followup)
event = true_time <= followup                          # True: the cancellation was seen; False: censored
print(f"customers {n}: cancellations seen {event.sum()}, still customers (censored) {(~event).sum()}")

def kaplan_meier(time, event):
    order = np.argsort(time)
    t, e = time[order], event[order]
    at_risk, s, curve = len(t), 1.0, []
    for ti, ei in zip(t, e):
        if ei:
            s *= 1 - 1 / at_risk                       # survival drops only at observed events
        curve.append((ti, s))
        at_risk -= 1                                   # censored customers leave the risk set without an event
    return curve

def median(curve):
    return next(t for t, s in curve if s <= 0.5)

km = kaplan_meier(time, event)
drop = kaplan_meier(time[event], event[event])         # ignore the customers who have not cancelled
as_event = kaplan_meier(time, np.ones(n, bool))        # pretend every customer cancelled at the end of follow-up
print(f"median months until cancelling: Kaplan-Meier {median(km):.1f}   (true {12 * np.log(2):.1f})")
print(f"   dropping censored customers {median(drop):.1f},   treating them as cancelled {median(as_event):.1f}")
print("ignoring censoring makes customers look less loyal:", median(drop) < median(km) - 1)`, after: `Half the customers are censored. Kaplan–Meier estimates a median of 7.6 months, close to the true 8.3 (the difference is sampling noise). Dropping the censored customers keeps only the ones who cancelled quickly enough to be seen, and halves the estimate to 3.3 months; treating them as cancellations is almost as bad. Both shortcuts make customers look far less loyal than they are, and the error grows the more data are censored.`, predict: {"type":"mcq","q":"Half of the customers have not cancelled yet. If you drop them and compute the median time to cancel from the rest, the estimate will be…","options":["About right","Too long","Too short","Impossible to compute"],"answer":2,"explain":"Only customers who cancelled within their observation window remain, and those are disproportionately the quick cancellers, so the median comes out far too short: 3.3 months instead of about 8.","expect":["ignoring censoring makes customers look less loyal: True"]} },
      { type: 'chart', title: "Survival curves with and without handling censoring", md: `The three estimates from the code above over the whole 18 months of follow-up, with the true survival curve of the simulation:`, after: `The Kaplan–Meier curve stays within five percentage points of the true curve over the whole follow-up: 36% still subscribed after a year, against a true 37%. Dropping the censored customers keeps only those who cancelled during the follow-up, so that curve falls to 4% after a year and to zero by 18 months, and counting the censored customers as cancellations is not much better. Neither shortcut can tell a customer who left from one who has not been watched long enough.`, chart: {"kind":"line","x":[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18],"points":false,"series":[{"name":"Kaplan–Meier","y":[1,0.9656,0.9161,0.8651,0.826,0.8013,0.762,0.7197,0.6906,0.6701,0.6461,0.627,0.5869,0.5509,0.5255,0.5022,0.4816,0.4706,0.4627,0.4379,0.42,0.4154,0.3776,0.3726,0.3621,0.3447,0.3378,0.3308,0.3236,0.3236,0.3141,0.3141,0.3141,0.2976,0.2728,0.2387,0.2387]},{"name":"dropping censored customers","y":[1,0.9317,0.8353,0.739,0.6667,0.6225,0.5542,0.4819,0.4337,0.4016,0.3655,0.3373,0.2811,0.2329,0.2008,0.1727,0.1486,0.1365,0.1285,0.1044,0.0884,0.0843,0.0522,0.0482,0.0402,0.0281,0.0241,0.0201,0.0161,0.0161,0.012,0.012,0.012,0.008,0.004,0,0]},{"name":"censored counted as cancelled","y":[1,0.948,0.872,0.802,0.746,0.7,0.654,0.606,0.556,0.516,0.48,0.45,0.398,0.36,0.318,0.298,0.272,0.252,0.23,0.202,0.186,0.18,0.152,0.144,0.13,0.114,0.098,0.094,0.09,0.074,0.062,0.052,0.046,0.032,0.022,0.008,0]},{"name":"true survival","y":[1,0.9592,0.92,0.8825,0.8465,0.8119,0.7788,0.747,0.7165,0.6873,0.6592,0.6323,0.6065,0.5818,0.558,0.5353,0.5134,0.4925,0.4724,0.4531,0.4346,0.4169,0.3998,0.3835,0.3679,0.3529,0.3385,0.3247,0.3114,0.2987,0.2865,0.2748,0.2636,0.2528,0.2425,0.2326,0.2231],"dash":true}],"xLabel":"months since joining","yLabel":"share still subscribed","yMin":0,"yMax":1} },
      { type: 'text', title: "Hazards, the Cox model and evaluating risk scores", md: `The **Cox proportional-hazards model** relates covariates to the hazard without assuming any shape for how the baseline risk changes over time:

$$ h(t | x) = h₀(t) · exp(β₁ x₁ + … + β_p x_p) $$
:::where
h(t | x) — the hazard at time t for a subject with covariates x
h₀(t) — the baseline hazard, left unspecified and shared by all subjects
β₁, …, β_p — the coefficients; exp(β_j) is the hazard ratio for a one-unit increase in x_j
:::

A hazard ratio of 2 means that, at any moment, a subject with that feature who has not had the event yet has twice the instantaneous risk. The model assumes these ratios stay constant over time (proportional hazards), which should be checked.

Risk scores are evaluated with the **concordance index** (C-index): among pairs of subjects whose order of events is known despite censoring, the share in which the subject with the higher predicted risk had the event first. 0.5 is random ranking, 1.0 perfect. Machine-learning extensions include random survival forests, gradient boosting with a Cox loss and neural survival models; in Python, lifelines and scikit-survival implement them, and statsmodels and scikit-learn fit GLMs.` },
      { type: 'check', question: {"type":"mcq","q":"A Cox model gives smokers a hazard ratio of 2 for heart disease. What does that mean?","options":["Smokers live half as long","At any time, smokers who are still healthy have about twice the instantaneous risk","Twice as many smokers will eventually get the disease","The model is twice as accurate for smokers"],"answer":1,"explain":"A hazard ratio compares instantaneous risks among those still at risk; it does not directly translate into lifetimes or lifetime probabilities."} }
    ],
    when: `- Counts per period (claims, tickets, visits): Poisson or negative binomial regression, with exposure as an offset.
- Positive, skewed amounts (claim sizes, durations): gamma regression with a log link.
- "How long until…?" questions where some subjects have not experienced the event yet: survival analysis.`,
    pitfalls: `- Fitting ordinary least squares to counts: it predicts negative values and treats a change from 0 to 1 like one from 100 to 101.
- Ignoring overdispersion: real counts usually vary more than a Poisson model allows, so its intervals are too narrow.
- Dropping censored subjects, or treating censoring as the event: both bias survival estimates, usually downwards.
- Reading a hazard ratio as a ratio of survival times: it compares instantaneous risks.`,
    takeaways: [
      'A GLM links a linear predictor to the mean of an exponential-family distribution: identity for normal, logit for Bernoulli, log for Poisson and gamma.',
      'In a log-link model, exp(coefficient) multiplies the expected count: a rate ratio.',
      'Survival data are censored: for many subjects we only know that the event had not happened yet.',
      'Kaplan–Meier estimates the survival curve using every subject while they are observed; dropping censored subjects makes survival look worse.',
      'The Cox model relates covariates to the hazard with exp(β) hazard ratios, and the concordance index measures how well risk scores rank event times.'
    ],
    quiz: [
      {"type":"mcq","q":"Which link function does a Poisson regression for counts usually use?","options":["Identity","Logit","Log","Square root"],"answer":2,"explain":"The log link keeps predicted rates positive and makes effects multiplicative."},
      {"type":"fill","q":"A Poisson coefficient is 0.693. By what factor does the expected count change per unit increase of that feature? (one decimal)","answer":["2.0","2"],"explain":"exp(0.693) ≈ 2.0: the expected count doubles."},
      {"type":"tf","q":"Overdispersion means the counts vary more than their mean, which a plain Poisson model cannot capture.","answer":true,"explain":"A Poisson variable's variance equals its mean; extra variance needs a negative binomial or quasi-Poisson model, or the intervals will be too narrow."},
      {"type":"mcq","q":"What is a censored observation in survival analysis?","options":["A data point removed for privacy","A subject for whom we only know that the event had not happened by the end of observation","An outlier","A missing feature"],"answer":1,"explain":"Censoring gives a lower bound on the time to the event; methods such as Kaplan–Meier use that information instead of discarding it."},
      {"type":"mcq","q":"A survival model has a concordance index of 0.5. What does that tell you?","options":["It ranks risks perfectly","It ranks risks no better than chance","Half the subjects had the event","It is well calibrated"],"answer":1,"explain":"The C-index is the share of comparable pairs ranked correctly; 0.5 is what random scores achieve."},
      {"type":"mcq","q":"Why include log(exposure) as an offset in a count model?","options":["To make the model nonlinear","Because counts depend on how long or how much each unit was observed, and the model should learn rates","To remove outliers","Because exposure is always zero"],"answer":1,"explain":"A policy observed for two years naturally has more claims than one observed for six months; the offset turns counts into rates."}
    ],
    problems: [
      {
        id: 'kaplan-meier', title: 'Kaplan–Meier Estimator', difficulty: 'medium',
        prompt: `Implement \`kaplan_meier(times, events)\`. \`times\` lists how long each subject was observed, and \`events[i]\` is 1 if subject i had the event at that time and 0 if it was censored (still event-free when observation stopped). Return a list of \`(time, survival)\` pairs, one for each **distinct time at which at least one event happened**, in increasing order.

At such a time t, with n subjects still at risk (observed until t or later) and d events at t, survival is multiplied by (1 − d / n). Subjects censored at t still count as at risk at t.

\`\`\`text
kaplan_meier([1, 2, 2, 3, 4, 5], [1, 1, 0, 1, 0, 1])
->  [(1, 0.8333), (2, 0.6667), (3, 0.4444), (5, 0.0)]      (rounded)
\`\`\``,
        starter: `
def kaplan_meier(times, events):
    # TODO: walk through the distinct event times in order
    pass
`,
        solution: `
def kaplan_meier(times, events):
    curve, s = [], 1.0
    for t in sorted(set(t for t, e in zip(times, events) if e)):
        at_risk = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e)
        s *= 1 - d / at_risk
        curve.append((t, s))
    return curve
`,
        hints: ['Only times with at least one event appear in the result.', 'At risk at time t: every subject whose observed time is t or later.', 'Keep a running product of (1 − d / n).'],
        explain: `Censored subjects contribute to the risk set for as long as they are watched, then drop out without an event. That is how Kaplan–Meier uses all the information without assuming anything about what happened after observation stopped.`,
        tests: [
          { call: '[(t, round(s, 4)) for t, s in kaplan_meier([1, 2, 2, 3, 4, 5], [1, 1, 0, 1, 0, 1])]', expect: '[(1, 0.8333), (2, 0.6667), (3, 0.4444), (5, 0.0)]' },
          { name: 'no censoring', call: '[(t, round(s, 4)) for t, s in kaplan_meier([3, 1, 2], [1, 1, 1])]', expect: '[(1, 0.6667), (2, 0.3333), (3, 0.0)]' },
          { name: 'ties among events', call: '[(t, round(s, 4)) for t, s in kaplan_meier([2, 2, 2, 5], [1, 1, 0, 1])]', expect: '[(2, 0.5), (5, 0.0)]' },
          { name: 'everyone censored', call: 'kaplan_meier([4, 6], [0, 0])', expect: '[]' },
          { name: 'random data vs reference', setup: 'import random\nrandom.seed(8)\nts = [random.randint(1, 20) for _ in range(40)]\nes = [random.random() < 0.6 for _ in range(40)]', call: '[(t, round(s, 6)) for t, s in kaplan_meier(ts, es)]', hidden: true }
        ]
      },
      {
        id: 'poisson-deviance', title: 'Poisson Deviance', difficulty: 'easy',
        prompt: `The **mean Poisson deviance** measures how well predicted rates \`mu\` explain observed counts \`y\`:

D = (2 / n) Σ [ y log(y / mu) − (y − mu) ],  where the term y log(y / mu) is taken as 0 when y = 0.

Implement \`poisson_deviance(y, mu)\` for two equal-length lists of numbers (all mu > 0).

\`\`\`text
poisson_deviance([2, 0], [2, 1])   ->  1.0       (0 for the first pair, 2 · 1 / 2 for the second)
\`\`\``,
        starter: `
import math

def poisson_deviance(y, mu):
    # TODO
    pass
`,
        solution: `
import math

def poisson_deviance(y, mu):
    total = 0.0
    for yi, mi in zip(y, mu):
        term = yi * math.log(yi / mi) if yi > 0 else 0.0
        total += term - (yi - mi)
    return 2 * total / len(y)
`,
        hints: ['Handle y = 0 separately: the y log(y / mu) term is zero there.', 'Divide the sum by the number of observations and multiply by 2.'],
        explain: `Deviance is the Poisson counterpart of squared error: it is zero for perfect predictions and penalizes relative rather than absolute mistakes, so predicting 1 for a count of 3 costs more than predicting 101 for a count of 103. Scikit-learn's mean_poisson_deviance computes the same quantity.`,
        tests: [
          { call: 'round(poisson_deviance([2, 0], [2, 1]), 6)', expect: '1.0' },
          { name: 'perfect predictions', call: 'round(poisson_deviance([1, 3, 5], [1, 3, 5]), 6)', expect: '0.0' },
          { name: 'relative errors matter', call: 'poisson_deviance([3], [1]) > poisson_deviance([103], [101])', expect: 'True' },
          { name: 'random counts vs reference', setup: 'import random\nrandom.seed(2)\nys = [random.randint(0, 8) for _ in range(30)]\nms = [random.uniform(0.5, 6) for _ in range(30)]', call: 'round(poisson_deviance(ys, ms), 6)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'scikit-learn, Generalized linear models (Poisson, gamma and Tweedie regression)', url: 'https://scikit-learn.org/stable/modules/linear_model.html' },
      { title: 'lifelines, survival analysis in Python (documentation)', url: 'https://lifelines.readthedocs.io/' }
    ]
  },

  // =====================================================================================
  {
    id: 'vision-3d', branch: 'spec', title: '3D Vision & Neural Rendering', short: '3D Vision', level: 3, minutes: 50, prereqs: ['cnn', 'vision-transformers'],
    tags: ['3d vision', 'depth estimation', 'stereo', 'disparity', 'camera model', 'pinhole camera', 'intrinsics', 'point cloud', 'pointnet', 'lidar', 'slam', 'nerf', 'neural radiance field', 'volume rendering', 'gaussian splatting', '3d generation'],
    summary: 'How machines recover and render the 3D world: camera models and projection, depth from stereo and why its error grows with distance, learning on point clouds with order-invariant networks, neural radiance fields and volume rendering, 3D Gaussian splatting for real-time novel views, and how 2D generative models are lifted to 3D.',
    why: `Robots, self-driving cars, augmented-reality headsets and film effects all need to know the shape of the world, not just what is in a picture. 3D vision recovers depth and geometry from cameras and sensors, and neural rendering has made it possible to capture a real scene from ordinary photos and fly through it photorealistically. These methods connect classical geometry with deep learning in a way few other areas do.`,
    objectives: [
      'Project 3D points into an image with the pinhole camera model, and compute depth from stereo disparity',
      'Explain why stereo depth error grows with the square of the distance',
      'Describe how point-cloud networks handle unordered points',
      'Explain volume rendering in neural radiance fields, and how Gaussian splatting renders in real time'
    ],
    sections: [
      { type: 'text', title: "From pixels to 3D", md: `A photograph flattens the world: everything along one line of sight lands on the same pixel. 3D vision recovers what was lost, or builds new views of a scene, using several kinds of representation:

- **depth maps**: one distance per pixel;
- **point clouds**: unordered sets of 3D points, the natural output of LiDAR scanners and depth cameras;
- **meshes**: surfaces made of triangles, the standard in graphics;
- **voxels**: a 3D grid of occupied cells;
- **implicit functions**: a network that answers "what is at this point in space?", such as a signed distance or a density and colour;
- **3D Gaussians**: millions of small, coloured, semi-transparent blobs.

Typical tasks are depth estimation, 3D object detection (for driving), reconstruction of objects and scenes, simultaneous localization and mapping (SLAM, for robots and headsets), rendering new views of a captured scene, and generating 3D assets from text or images.` },
      { type: 'text', title: "Cameras, projection and stereo", md: `The **pinhole camera model** maps a 3D point to a pixel by dividing by its depth:

$$ u = f · X / Z + c_x          v = f · Y / Z + c_y $$
:::where
X, Y, Z — the point's coordinates in the camera's frame; Z is the depth along the viewing direction
f — the focal length, in pixels
c_x, c_y — the pixel where the optical axis meets the image, usually near its centre
u, v — the pixel coordinates of the projected point
:::

f, c_x and c_y are the camera's *intrinsics*; its position and orientation in the world are its *extrinsics*. Calibration estimates both.

Dividing by Z throws the depth away, so it has to come from somewhere else. **Stereo** uses two cameras a known distance apart: the same point appears shifted horizontally between the two images, and that shift, the *disparity*, reveals the depth:

$$ Z = f · B / d          ΔZ ≈ Z² · Δd / (f · B) $$
:::where
B — the baseline: the distance between the two cameras
d — the disparity: how many pixels the point shifts between the left and right images
Δd — an error in measuring the disparity, typically a fraction of a pixel to a pixel
ΔZ — the resulting error in depth
:::

The second formula is the important one: a fixed matching error causes a depth error that grows with the **square** of the distance.` },
      { type: 'code', title: "Projection and stereo depth", code: `import numpy as np
f, cx, cy = 800.0, 640.0, 360.0                        # focal length and image centre in pixels (a 1280 x 720 camera)
def project(P):                                        # the pinhole camera: divide by depth
    return np.column_stack([f * P[:, 0] / P[:, 2] + cx, f * P[:, 1] / P[:, 2] + cy])

cube = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (4.0, 5.0)])   # a 1 m cube 4-5 m away
print("cube corners in the image (pixels):", project(cube).round(0).astype(int).tolist())

baseline = 0.12                                        # two cameras 12 cm apart
depth = lambda disparity: f * baseline / disparity
for Z in (2, 5, 10, 20, 40):
    d = f * baseline / Z                               # how far the point shifts between the two images
    print(f"depth {Z:2d} m: disparity {d:5.2f} px, a one-pixel matching error moves it by {depth(d - 1) - Z:5.2f} m")
e10, e20 = depth(f * baseline / 10 - 1) - 10, depth(f * baseline / 20 - 1) - 20
print("twice as far, about four times the depth error:", bool(3.5 < e20 / e10 < 5))`, after: `The far face of the cube projects to a smaller square than the near face, which is perspective. For stereo, a one-pixel mistake costs 4 cm at 2 m but more than a metre at 10 m and almost 30 m at 40 m. That is why self-driving cars combine cameras with LiDAR or radar for long range, why stereo rigs for distant objects use wide baselines, and why learned depth networks are judged by their relative rather than absolute error.`, predict: {"type":"mcq","q":"With this stereo camera, a one-pixel matching error causes about 1.2 m of depth error at 10 m. Roughly how large is the error at 20 m?","options":["About the same, 1.2 m","About 2.4 m, twice as large","About 5 m, four times as large","About 0.6 m, half as large"],"answer":2,"explain":"Depth error grows roughly with the square of the distance (ΔZ ≈ Z²·Δd / (f·B)); here it is 5.3 m at 20 m.","expect":["twice as far, about four times the depth error: True"]} },
      { type: 'chart', title: "Stereo depth error against distance", md: `The formula from the code above (focal length 800 pixels) for the 12 cm rig and for a wider 50 cm rig:`, after: `The error grows roughly with the square of the distance and explodes as the disparity approaches one pixel: at 50 m the 12 cm rig can be off by more than 50 m. Four times the baseline cuts the error by four times or more at every distance (1.16 m against 0.26 m at 10 m), which is why long-range stereo uses wide rigs and why cars add LiDAR or radar.`, chart: {"kind":"line","x":[1,2,3,5,7,10,15,20,25,30,35,40,45,50],"yLog":true,"series":[{"name":"cameras 12 cm apart","y":[0.01053,0.04255,0.09677,0.2747,0.5506,1.163,2.778,5.263,8.803,13.64,20.08,28.57,39.71,54.35]},{"name":"cameras 50 cm apart","y":[0.002506,0.01005,0.02267,0.06329,0.1247,0.2564,0.5844,1.053,1.667,2.432,3.356,4.444,5.704,7.143]}],"xLabel":"distance to the point (m)","yLabel":"error from a 1-pixel mistake (m, log scale)"} },
      { type: 'text', title: "Learning on point clouds", md: `LiDAR scanners and depth cameras produce **point clouds**: sets of (x, y, z) points, often with intensity or colour. Two properties make them awkward for ordinary networks. They are **unordered**, so the same scene can be listed in any order, and **irregular**, with no grid for a convolution to slide over.

- **PointNet** (2017) applies the same small network to every point and then combines the results with a symmetric function, the maximum over points, so the order cannot matter. PointNet++ adds the same idea on local neighbourhoods, to capture fine structure.
- **Voxel and sparse-convolution networks** snap the points to a 3D grid and convolve only over occupied cells, which works well for large outdoor scenes.
- **Point transformers** use attention between nearby points, which is also order-independent as long as positions enter as features rather than as a sequence index.

These networks detect cars and pedestrians for autonomous driving, segment rooms for robots, and classify scanned parts in factories.` },
      { type: 'code', title: "Order does not matter", code: `import numpy as np
rng = np.random.default_rng(0)
points = rng.normal(size=(100, 3))                     # a point cloud: 100 points, each with x, y and z
W = rng.normal(size=(3, 16))                           # one small layer applied to every point
features = lambda p: np.maximum(p @ W, 0)             # 16 features per point
pooled = lambda p: features(p).max(axis=0)            # PointNet's idea: a symmetric function (max) over the points
flat = lambda p: features(p).reshape(-1)              # a naive alternative that depends on the order of the points
shuffled = points[rng.permutation(len(points))]
print("max-pooled features unchanged when the points are reordered:", bool(np.allclose(pooled(points), pooled(shuffled))))
print("flattened features unchanged when the points are reordered:", bool(np.allclose(flat(points), flat(shuffled))))`, after: `Taking the maximum over points gives the same global description however the points are listed, while flattening the per-point features into one long vector, as a naive network would, changes completely. Building the symmetry into the architecture means the network never has to learn it from examples (see [Geometric & Equivariant Deep Learning](#/concept/geometric-dl)).`, predict: {"type":"mcq","q":"The same 100 points are listed in a new random order. Which representation stays exactly the same?","options":["The flattened per-point features","The max-pooled features","Both","Neither"],"answer":1,"explain":"The maximum over a set does not depend on the order of its elements; a flattened list does.","expect":["max-pooled features unchanged when the points are reordered: True","flattened features unchanged when the points are reordered: False"]} },
      { type: 'text', title: "Neural radiance fields", md: `A **neural radiance field** (NeRF, 2020) stores a whole scene inside a neural network. Given a 3D position and a viewing direction, the network returns a **density** (how much matter is there) and a **colour**. To render a pixel, the renderer marches along the camera ray through that pixel, samples the network at many points, and blends them with **volume rendering**:

$$ C = Σ_i T_i · α_i · c_i          α_i = 1 − exp(−σ_i δ_i)          T_i = Π_{j<i} (1 − α_j) $$
:::where
C — the colour of the pixel
c_i — the colour the network predicts at sample i along the ray
σ_i — the density at sample i
δ_i — the distance between sample i and the next one
α_i — the chance that the ray stops within that segment
T_i — the transmittance: the chance that the ray travels unblocked to sample i
:::

Because every step is differentiable, the network can be trained from ordinary photographs with known camera poses (usually estimated with structure-from-motion software): render pixels, compare them with the photos, backpropagate. The original NeRF needed hours of training and seconds per frame; hash-grid encodings (Instant-NGP, 2022) cut training to minutes.` },
      { type: 'code', title: "Volume rendering along one ray", code: `import numpy as np
# Samples along one camera ray, from the camera outwards: density and brightness at each
sigma = np.array([0.0, 0.0, 0.1, 8.0, 8.0, 0.5, 0.0, 3.0])
color = np.array([0.0, 0.0, 0.2, 0.9, 0.9, 0.4, 0.0, 0.1])
delta = np.full(len(sigma), 0.25)                      # distance between neighbouring samples
alpha = 1 - np.exp(-sigma * delta)                     # chance that the ray stops within each segment
T = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]])   # transmittance: chance that the ray gets this far
w = T * alpha                                          # how much each sample contributes to the pixel
for i in range(len(sigma)):
    print(f"sample {i}: density {sigma[i]:3.1f}   ray reaches it {T[i]:.3f}   weight {w[i]:.3f}")
print(f"pixel brightness {np.sum(w * color):.3f}, opacity {w.sum():.3f}")
print("samples behind the dense surface hardly count:", bool(w[5:].sum() < 0.05))`, after: `The first dense sample takes 84% of the weight and the second most of the rest, so the pixel shows the surface the ray hits first. Samples behind it contribute about 1% together, even the fairly dense one at the end: the ray almost never gets that far. Occlusion emerges naturally from the transmittance, with no explicit rule for which surface is in front.`, predict: {"type":"mcq","q":"A dense, nearly opaque region sits early on the ray. How much do the samples behind it contribute to the pixel?","options":["As much as the front samples","About half","Almost nothing","Exactly nothing"],"answer":2,"explain":"The transmittance behind a dense region is tiny, so later samples get almost no weight (about 1% here), though never exactly zero.","expect":["samples behind the dense surface hardly count: True"]} },
      { type: 'text', title: "3D Gaussian splatting and 3D generation", md: `**3D Gaussian splatting** (2023) represents a scene with millions of small 3D Gaussians, each with a position, a shape (covariance), an opacity and a view-dependent colour. Instead of querying a network along every ray, the renderer projects ("splats") the Gaussians onto the image, sorts them by depth and blends them front to back, which graphics cards do extremely fast. Trained from photos in the same way as NeRF, it renders captured scenes in real time at high quality, at the cost of large files; it is now widely used for virtual tours, film and digital twins.

**Generating 3D content** is harder than generating images because 3D training data are scarce. Approaches include using a 2D image diffusion model as a critic that a 3D representation is optimized against (score distillation, DreamFusion, 2022), diffusion models that generate several consistent views of an object at once, and feed-forward reconstruction models that turn one or a few images into a 3D asset in seconds.` },
      { type: 'check', question: {"type":"tf","q":"From a single ordinary photograph, the absolute distance to an object can be computed exactly with the pinhole camera model.","answer":false,"explain":"Projection divides by depth, so every point along a line of sight lands on the same pixel. Depth needs a second view, a depth sensor, or learned priors about object sizes."} }
    ],
    when: `- Measuring distances and shapes: robotics, autonomous driving, inspection, construction.
- Capturing real places or objects for viewing from new angles: virtual tours, film, games, digital twins.
- Prefer 2D methods when the task does not depend on geometry, such as classifying what an image shows.`,
    pitfalls: `- Trusting stereo depth far away: the error grows with the square of the distance.
- Feeding a point cloud to a model that depends on the order of the points.
- Expecting a radiance field to work from photos without accurate camera poses, or from too few viewpoints.
- Forgetting scale: from a single image, depth is only known up to an unknown scale factor.`,
    takeaways: [
      'A pinhole camera divides by depth: u = f·X/Z + c_x, so distant objects shrink and depth is lost in a single image.',
      'Stereo recovers depth as Z = f·B/d, and a fixed disparity error causes a depth error that grows with Z².',
      'Point-cloud networks use symmetric functions such as max pooling, so the order of the points does not matter.',
      'NeRF stores a scene in a network and renders pixels by integrating density and colour along rays.',
      '3D Gaussian splatting represents a scene with millions of small Gaussians that can be rendered in real time.'
    ],
    quiz: [
      {"type":"fill","q":"A camera has focal length 1,000 pixels. A point lies 2 m to the right of the optical axis at a depth of 10 m. How many pixels from the image centre does it appear? (f · X / Z)","answer":["200"],"explain":"1,000 × 2 / 10 = 200 pixels."},
      {"type":"mcq","q":"How does a stereo camera recover depth?","options":["From the colour of objects","From how far a point shifts between the two images (the disparity)","From the image resolution","From the focal length alone"],"answer":1,"explain":"Z = f · B / d: nearby points shift a lot between the two views, distant points very little."},
      {"type":"mcq","q":"Why does PointNet use max pooling over points?","options":["To make the network smaller","So that its output does not depend on the order in which points are listed","To increase the resolution","To remove noise"],"answer":1,"explain":"A symmetric function such as the maximum ignores the order of its inputs, which suits unordered point sets."},
      {"type":"mcq","q":"What does a NeRF network predict for a 3D position and viewing direction?","options":["A class label","A density and a colour","A depth map","A mesh"],"answer":1,"explain":"Volume rendering combines densities and colours along each camera ray to produce the pixel."},
      {"type":"tf","q":"3D Gaussian splatting renders by projecting and blending Gaussians instead of evaluating a network at many points along every ray, which makes it fast enough for real time.","answer":true,"explain":"Rasterizing sorted Gaussians maps well onto graphics hardware, giving real-time rendering of captured scenes."},
      {"type":"mcq","q":"Why do self-driving cars often add LiDAR or radar to cameras?","options":["Cameras cannot see colour","Stereo depth error grows quickly with distance, while LiDAR and radar measure range directly","LiDAR is cheaper than cameras","Cameras do not work during the day"],"answer":1,"explain":"Depth from cameras degrades roughly with the square of the distance; active sensors measure distance far away more reliably."}
    ],
    problems: [
      {
        id: 'pinhole-projection', title: 'Project Points into an Image', difficulty: 'easy',
        prompt: `A pinhole camera at the origin looks along the z axis. Implement \`project(points, f, cx, cy)\` that maps each 3D point (X, Y, Z), with Z > 0, to its pixel coordinates

u = f · X / Z + cx,   v = f · Y / Z + cy

and returns a list of \`(u, v)\` tuples.

\`\`\`text
project([(0, 0, 5), (1, 0, 2)], 800, 640, 360)   ->  [(640.0, 360.0), (1040.0, 360.0)]
\`\`\``,
        starter: `
def project(points, f, cx, cy):
    # TODO
    pass
`,
        solution: `
def project(points, f, cx, cy):
    return [(f * X / Z + cx, f * Y / Z + cy) for X, Y, Z in points]
`,
        hints: ['Divide X and Y by the depth Z, multiply by the focal length, then add the image centre.'],
        explain: `Dividing by Z is what makes distant objects small, and it is also why a single image loses depth: every point along a ray through the camera centre lands on the same pixel. Depth has to come from a second view, a depth sensor, or learned priors.`,
        tests: [
          { call: 'project([(0, 0, 5), (1, 0, 2)], 800, 640, 360)', expect: '[(640.0, 360.0), (1040.0, 360.0)]' },
          { name: 'twice as far, half the offset', call: 'project([(1, 1, 4), (1, 1, 8)], 400, 0, 0)', expect: '[(100.0, 100.0), (50.0, 50.0)]' },
          { name: 'points behind each other share a pixel', call: 'project([(1, 2, 4), (2, 4, 8)], 500, 320, 240)', expect: '[(445.0, 490.0), (445.0, 490.0)]' },
          { name: 'random points vs reference', setup: 'import random\nrandom.seed(4)\npts = [(random.uniform(-2, 2), random.uniform(-1, 1), random.uniform(1, 9)) for _ in range(10)]', call: '[(round(u, 4), round(v, 4)) for u, v in project(pts, 700, 320, 240)]', hidden: true }
        ]
      },
      {
        id: 'volume-render', title: 'Volume Rendering along a Ray', difficulty: 'medium',
        prompt: `Neural radiance fields compute a pixel by combining samples along a camera ray. For sample i with density \`sigmas[i]\` and colour \`colors[i]\` (one number here), and spacing \`delta\` between samples:

- α_i = 1 − exp(−σ_i · delta): the chance that the ray stops in that segment;
- T_i = the product of (1 − α_j) over all earlier samples j < i (T_0 = 1): the chance that the ray reaches sample i;
- the pixel colour is Σ T_i · α_i · colour_i, and the opacity is Σ T_i · α_i.

Implement \`render(sigmas, colors, delta)\` returning \`(colour, opacity)\`.`,
        starter: `
import math

def render(sigmas, colors, delta):
    # TODO: accumulate front to back
    pass
`,
        solution: `
import math

def render(sigmas, colors, delta):
    T, colour, opacity = 1.0, 0.0, 0.0
    for s, c in zip(sigmas, colors):
        a = 1 - math.exp(-s * delta)
        colour += T * a * c
        opacity += T * a
        T *= 1 - a
    return colour, opacity
`,
        hints: ['Walk from the camera outwards, keeping the transmittance T.', 'Add T · α · colour, then multiply T by (1 − α).'],
        explain: `Because the weights T_i · α_i are differentiable in the densities and colours, a network that predicts σ and colour at any 3D point can be trained from photos alone: render the pixels, compare them with the real ones, and backpropagate. That is the core of NeRF.`,
        tests: [
          { name: 'empty space', call: 'render([0, 0, 0], [1, 1, 1], 0.5)', expect: '(0.0, 0.0)' },
          { name: 'an opaque wall hides what is behind it', call: '[round(x, 4) for x in render([50, 50], [0.2, 0.9], 1.0)]', expect: '[0.2, 1.0]' },
          { name: 'a half-transparent sample', call: '[round(x, 4) for x in render([math.log(2)], [1.0], 1.0)]', setup: 'import math', expect: '[0.5, 0.5]' },
          { name: 'a mixed ray vs reference', call: '[round(x, 6) for x in render([0.0, 0.1, 8.0, 8.0, 0.5, 3.0], [0.0, 0.2, 0.9, 0.9, 0.4, 0.1], 0.25)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields (2020)', url: 'https://arxiv.org/abs/2003.08934' },
      { title: 'Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023)', url: 'https://arxiv.org/abs/2308.04079' },
      { title: 'Qi et al., PointNet: Deep Learning on Point Sets (2016)', url: 'https://arxiv.org/abs/1612.00593' }
    ]
  },

  // =====================================================================================
  {
    id: 'ai-for-science', branch: 'spec', title: 'AI for Science: Proteins, Molecules and Weather', short: 'AI for Science', level: 3, minutes: 50, prereqs: ['gnn', 'transformer-block'],
    tags: ['ai for science', 'protein structure', 'alphafold', 'multiple sequence alignment', 'co-evolution', 'plddt', 'protein design', 'molecules', 'materials', 'weather forecasting', 'graphcast', 'ensemble forecast', 'chaos', 'reanalysis', 'scientific validation'],
    summary: 'How machine learning became a scientific instrument: protein structure prediction from evolutionary signals (AlphaFold) and protein design, molecules and materials as graphs, data-driven weather forecasting and why ensembles matter in chaotic systems, and the standards of validation that scientific use demands.',
    why: `Some of the most consequential results of modern AI are scientific: predicted structures for nearly every known protein, weather forecasts that rival the best physical models at a fraction of the computing cost, and millions of candidate materials. Science also asks things of models that consumer applications rarely do: respecting physical laws, quantifying uncertainty, and surviving real experiments. This lesson shows the ideas behind these systems and the habits needed to use them responsibly.`,
    objectives: [
      'Explain how co-evolution in sequence alignments reveals which parts of a protein touch',
      'Describe, at a high level, how AlphaFold-style models predict and score structures',
      'Explain why chaotic systems such as the weather need ensemble forecasts',
      'Apply scientific standards to ML results: proper splits, uncertainty and experimental validation'
    ],
    sections: [
      { type: 'text', title: "Why science is different", md: `Some of the most consequential uses of machine learning are scientific:

- **Protein structures**: AlphaFold 2 (2020) predicted protein structures from sequences with accuracy close to experiments, and its database now holds predicted structures for over 200 million proteins. The 2024 Nobel Prize in Chemistry went to Demis Hassabis and John Jumper for protein structure prediction, and to David Baker for computational protein design.
- **Weather**: learned models trained on decades of past weather now rival the best physics-based forecasts at a tiny fraction of the computing cost, and weather services have begun running them operationally.
- **Materials and molecules**: models screen millions of candidate crystals and drug-like molecules before anything is synthesized.
- **Mathematics and control**: systems have solved olympiad geometry and proof problems and controlled the plasma in a fusion reactor.

Science asks more of a model than most applications do. Predictions must respect physical laws and symmetries, come with honest uncertainty, generalize to things never seen before (a new protein family, an unusual storm), and ultimately survive an experiment. Good benchmarks are hard to build, because similar examples leak easily between training and test sets.` },
      { type: 'text', title: "Proteins: from sequence to structure", md: `A protein is a chain of amino acids (20 kinds) that folds into a 3D shape, and the shape determines what it does. Experiments that determine structures take months; sequences are cheap. The key signal for predicting structure is **evolution**: across species, the same protein appears with many variations, and a **multiple sequence alignment** (MSA) lines those variants up position by position. When two positions touch in the folded protein, a mutation at one is often compensated by a mutation at the other, so the two columns **co-evolve**.

AlphaFold 2 combines this evolutionary signal with a learned sense of geometry:

- a transformer-style trunk (the Evoformer) repeatedly exchanges information between a representation of the MSA and a representation of every pair of positions;
- a structure module turns the pair representation into 3D coordinates for every atom, using attention that is unaffected by rotating or moving the whole protein;
- the whole network runs several times on its own output (recycling), and reports a per-residue confidence score, **pLDDT**, from 0 to 100.

Since then, AlphaFold 3 (2024) predicts complexes of proteins with DNA, RNA and small molecules using a diffusion module, protein language models such as ESM predict structure from a single sequence without an alignment, and design tools such as RFdiffusion and ProteinMPNN invert the problem: generating new proteins with a desired shape or function.` },
      { type: 'code', title: "Co-evolution reveals which positions touch", md: `A synthetic alignment of 400 sequences of a 12-residue protein. Most positions mutate independently, but positions 3 and 9 touch: one must carry a positive charge when the other carries a negative one, so they change together:`, code: `import numpy as np
rng = np.random.default_rng(0)
L, N = 12, 400                                         # positions in the protein, sequences in the alignment
alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
msa = rng.choice(alphabet, size=(N, L))                # most positions vary independently
partner = {"K": "E", "E": "K", "R": "D", "D": "R"}     # positions 3 and 9 touch: opposite charges must face each other
msa[:, 3] = rng.choice(list(partner), N)
msa[:, 9] = [partner[a] if rng.random() < 0.9 else rng.choice(alphabet) for a in msa[:, 3]]

def mutual_information(a, b):
    mi = 0.0
    for x in np.unique(a):
        ax = a == x
        for y in np.unique(b):
            pxy = np.mean(ax & (b == y))
            if pxy > 0:
                mi += pxy * np.log(pxy / (ax.mean() * np.mean(b == y)))
    return mi

scores = {(i, j): mutual_information(msa[:, i], msa[:, j]) for i in range(L) for j in range(i + 1, L)}
ranked = sorted(scores, key=scores.get, reverse=True)
for i, j in ranked[:3]:
    print(f"positions {i:2d} and {j:2d}: mutual information {scores[(i, j)]:.3f}")
print(f"median over all pairs: {np.median(list(scores.values())):.3f}")
print("the co-evolving pair stands out:", ranked[0] == (3, 9))`, after: `The touching pair stands out clearly, with more than twice the mutual information of any other pair. Note that unrelated pairs still score about 0.5, not zero: with 20 letters and only 400 sequences, chance coincidences look like information. Real contact prediction had to correct for this and for indirect correlations (A with B and B with C makes A look related to C), which led from simple statistics to global statistical models and finally to deep networks that learn from the whole alignment at once.`, predict: {"type":"mcq","q":"Positions 3 and 9 must keep opposite charges, so they mutate together. Which pair will show the highest mutual information?","options":["A random pair of positions","Positions 3 and 9","All pairs score the same","Positions 0 and 1, because they are neighbours in the sequence"],"answer":1,"explain":"Co-evolving positions share information: knowing one predicts the other. Here the pair scores 1.26 against about 0.5 for unrelated pairs.","expect":["the co-evolving pair stands out: True"]} },
      { type: 'text', title: "Molecules and materials as graphs", md: `A molecule is naturally a graph: atoms are nodes, bonds are edges, and 3D coordinates add geometry. **Graph neural networks** (see [Graph Neural Networks](#/concept/gnn)) predict properties such as solubility, toxicity or binding strength from that graph, and networks that respect rotations and translations (see [Geometric & Equivariant Deep Learning](#/concept/geometric-dl)) learn interatomic forces accurately enough to replace expensive quantum-chemistry calculations in simulations. Generative models propose new molecules, either as 3D coordinates with diffusion models or as text strings with language models.

Materials science uses the same tools on crystals. Google DeepMind's GNoME project (2023) predicted about 2.2 million new crystal structures, roughly 380,000 of them judged stable. Such results also show the limits: predicted stability is not the same as being possible to synthesize, and claims of novelty need careful checking against what is already known.` },
      { type: 'text', title: "Weather and climate", md: `Traditional **numerical weather prediction** solves the equations of the atmosphere on supercomputers. Learned models instead train on **reanalysis** data such as ERA5, decades of past weather reconstructed consistently on a global grid, and learn to step the atmosphere forward six hours at a time.

- **GraphCast** (Google DeepMind, 2023) runs a graph neural network on a mesh around the globe and produces a 10-day forecast in under a minute on a single accelerator; it beat the leading physics model on most of the variables evaluated.
- **GenCast** (2024) generates whole *ensembles* of forecasts with a diffusion model, to express uncertainty.
- The European Centre for Medium-Range Weather Forecasts (ECMWF) began running its own learned model, AIFS, operationally alongside its physics model in 2025.

The atmosphere is **chaotic**: tiny errors in the starting state grow until a single forecast is no better than a guess. Weather services therefore run **ensembles**, many forecasts from slightly different starting states. The spread of the ensemble shows how uncertain the forecast is, and its mean is more accurate than any single member far ahead.` },
      { type: 'code', title: "Chaos and ensembles", md: `The Lorenz system, a three-variable model of convection that was the original example of chaos, standing in for the atmosphere. For 100 different situations, one forecast and a 50-member ensemble start from slightly wrong measurements:`, code: `import numpy as np
rng = np.random.default_rng(0)
def step(s, dt=0.01, sig=10.0, rho=28.0, beta=8 / 3):  # the Lorenz system, a classic chaotic model of convection
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return s + dt * np.stack([sig * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)

cases, members = 100, 50
truth = rng.normal(size=(cases, 3)) + [1.0, 1.0, 20.0]
for _ in range(1000):                                  # let each case settle onto the attractor
    truth = step(truth)
start = truth + rng.normal(0, 0.1, truth.shape)        # what we measure: the true state plus small errors
single = start.copy()
ensemble = start[:, None, :] + rng.normal(0, 0.1, (cases, members, 3))
print("lead time   one forecast   ensemble mean   ensemble spread")
for k in range(1, 2001):
    truth, single, ensemble = step(truth), step(single), step(ensemble)
    if k % 250 == 0:
        e1 = np.linalg.norm(single - truth, axis=1).mean()
        em = np.linalg.norm(ensemble.mean(axis=1) - truth, axis=1).mean()
        spread = np.linalg.norm(ensemble.std(axis=1), axis=1).mean()
        print(f"{k * 0.01:8.1f} {e1:13.2f} {em:15.2f} {spread:16.2f}")
print("far ahead, the ensemble mean errs less than one forecast:", bool(em < 0.85 * e1))`, after: `At short lead times every forecast is good. Then errors of 0.1 in the starting state grow until a single forecast is about as far from the truth as a random state on the attractor. The ensemble mean settles about 30% lower, because averaging many equally plausible futures cancels part of their disagreement, and the ensemble spread grows to match the error, so the forecast honestly reports its own uncertainty.`, predict: {"type":"mcq","q":"Far ahead, after the errors have grown, which has the smaller average error?","options":["A single forecast started from the best measurement","The mean of a 50-member ensemble","They are exactly equal","Neither can be computed"],"answer":1,"explain":"Once the forecasts have spread out, averaging them cancels part of the random error; the ensemble mean ends up about 30% closer to the truth here.","expect":["far ahead, the ensemble mean errs less than one forecast: True"]} },
      { type: 'chart', title: "Forecast error against lead time", md: `The forecasts from the code above, measured every 25 steps instead of every 250:`, after: `Errors that start at about 0.2 grow quickly, and by a lead time of about 5 a single forecast is as far from the truth as a random point on the attractor and stops getting worse. The ensemble mean levels off 20 to 30% lower. The spread follows the ensemble mean's error at every lead time, which is what makes an ensemble useful: it says how far to trust each forecast.`, chart: {"kind":"line","x":[0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5,11.75,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,19.75,20],"points":false,"series":[{"name":"one forecast","y":[0.2288,0.4602,0.7006,1.177,1.698,1.684,1.815,1.967,2.927,3.268,5.929,6.075,5.56,8.181,9.666,9.208,11.55,11.12,12.74,16.38,13.44,16.04,15.93,13.74,16.52,15.26,13.28,15.16,16.27,16.55,17.7,17.45,17.19,18.15,16.67,18.4,17.11,14.86,17.52,17.87,15.81,18.27,15.83,15.34,18.11,17.56,18.17,18.53,18.33,18.19,18.51,17.28,17.88,16.58,17.73,19.34,16.72,18.43,19.22,17.7,19.27,18.94,17.15,19.34,19.61,18.86,19.36,17.59,17.81,19.14,18.83,18.32,18.78,17.8,18.1,18.73,18.04,18.6,18.57,17.97]},{"name":"ensemble mean","y":[0.2317,0.4606,0.6952,0.9,1.3,1.483,1.611,1.699,2.693,3.065,4.94,5.685,5.504,7.801,8.252,9.045,10.57,9.224,11.56,13.26,10.74,13.24,13.66,11.57,13.6,12.86,11.42,13.9,11.91,12.28,14.1,13.09,11.78,13.5,12.73,13.1,14.32,11.94,13.5,13.9,12.62,14.2,13.23,12.38,14.35,12.95,13.61,13.5,13.41,12.93,14.78,12.51,13.37,13.58,12.72,13.71,11.84,13.29,14.91,12.59,13.61,14.17,12.31,13.61,13.71,12.95,13.54,13.14,12.39,14.42,12.65,12.49,13.87,13.62,13.24,13.92,13.18,13.42,13.94,12.63]},{"name":"ensemble spread","y":[0.2396,0.4766,0.9555,1.177,1.389,2.026,2.148,2.502,3.538,4.082,5.375,5.875,6.456,8.561,9.12,10.04,12.24,11.3,12.56,13.39,12.35,13.77,13.66,12.71,14.27,13.55,13.29,14.45,13.49,13.59,14.41,13.52,14.08,14.19,13.4,14.17,13.98,13.5,14.42,13.91,13.95,14.43,13.71,14.17,14.29,13.81,14.4,14.44,13.74,14.42,14.05,13.94,14.41,13.94,14.09,14.38,13.96,14.15,14.39,13.96,14.33,14.35,14.07,14.45,14.13,14.16,14.63,14.08,14.36,14.49,14.12,14.24,14.34,14.13,14.27,14.18,14.21,14.38,14.2,14.29],"dash":true}],"xLabel":"lead time (model time units)","yLabel":"average error","yMin":0} },
      { type: 'text', title: "Doing science with machine learning", md: `- **Split by similarity, not at random.** Proteins in the test set should not be close relatives of training proteins, molecules should come from new scaffolds, and weather models should be tested on later years than they were trained on. Random splits make models look far better than they are.
- **Report uncertainty**, and check that it is calibrated: confidence scores such as pLDDT are only useful because they were validated against experiments.
- **Compare with strong baselines**, including physics-based methods, not only with older machine-learning models.
- **Validate prospectively.** The strongest evidence is a prediction made before the experiment and then confirmed in the lab or by the next season's weather.
- **Respect the physics.** Conservation laws, symmetries and units are constraints a model can violate silently; build them in where possible and test for them where not.` },
      { type: 'check', question: {"type":"mcq","q":"A protein-property model is evaluated with a random train/test split and scores very well. What is the most likely problem?","options":["The model is too small","Close relatives of test proteins are in the training set, so the score overstates performance on genuinely new proteins","Random splits always underestimate performance","Proteins cannot be predicted"],"answer":1,"explain":"Similar sequences leak information between the sets. Splitting by sequence similarity gives an honest estimate for new protein families."} }
    ],
    when: `- Expensive experiments or simulations limit progress, and good data from past experiments exist.
- Screening huge spaces of candidates (molecules, materials, protein designs) before testing the best ones in the lab.
- Fast surrogate models for slow physical simulations, checked against the simulator.`,
    pitfalls: `- Random train/test splits with near-duplicates: similar proteins or overlapping time periods leak between them and inflate results.
- Treating a model's confident prediction as a measurement: confidence scores must be checked, and key results confirmed experimentally.
- Single forecasts of chaotic systems far ahead: only ensembles give honest uncertainty.
- Ignoring physical constraints such as conservation laws and symmetries, which models can violate silently.`,
    takeaways: [
      'Residues that touch in a folded protein mutate together across evolution, which sequence alignments reveal.',
      'AlphaFold combines evolutionary and pairwise information in a transformer and outputs 3D coordinates with per-residue confidence (pLDDT).',
      'Molecules and materials are naturally graphs, and models that respect their symmetries learn from less data.',
      'In chaotic systems tiny initial errors grow; ensemble forecasts express that uncertainty, and their mean is more accurate far ahead.',
      'Scientific ML needs leakage-free splits, calibrated uncertainty and experimental validation.'
    ],
    quiz: [
      {"type":"mcq","q":"What evolutionary signal reveals which parts of a protein touch?","options":["The length of the protein","Co-evolution: positions that touch tend to mutate together across species","The number of species","The protein's name"],"answer":1,"explain":"Compensating mutations at contacting positions leave correlated columns in a multiple sequence alignment."},
      {"type":"mcq","q":"What does AlphaFold's pLDDT score tell you?","options":["The protein's function","How confident the model is in the predicted structure of each residue","The protein's size","The speed of folding"],"answer":1,"explain":"pLDDT is a per-residue confidence from 0 to 100; low values often mark flexible or disordered regions."},
      {"type":"tf","q":"Because the atmosphere is chaotic, small errors in the starting state eventually grow until a single forecast is no better than a guess.","answer":true,"explain":"That is why forecasters run ensembles: the spread shows the uncertainty and the mean is more accurate far ahead."},
      {"type":"mcq","q":"What is reanalysis data, such as ERA5?","options":["Raw satellite images","A consistent reconstruction of past weather on a global grid, combining observations with a physical model","Future climate projections","A list of extreme events"],"answer":1,"explain":"Learned weather models are trained on decades of reanalysis, which itself relies on physics-based data assimilation."},
      {"type":"mcq","q":"A model predicts that a new crystal is stable. What should happen before claiming a discovery?","options":["Nothing: the prediction is enough","Check novelty against known materials and try to synthesize and characterize it","Train a bigger model","Publish the prediction as a measurement"],"answer":1,"explain":"Predicted stability does not guarantee that a material can be made, or that it is new; experiments and careful literature checks decide."},
      {"type":"fill","q":"An ensemble's members are 3.0, 5.0 and 4.0 °C, and the truth is 4.0 °C. What is the error of the ensemble mean, in °C?","answer":["0","0.0"],"explain":"The mean is 4.0, exactly the truth, although each member except one was off by 1 °C."}
    ],
    problems: [
      {
        id: 'column-mutual-information', title: 'Mutual Information between Two Columns', difficulty: 'medium',
        prompt: `In a multiple sequence alignment, positions that co-evolve share information. Implement \`mutual_information(a, b)\` for two equal-length sequences of symbols (strings or lists), using natural logarithms:

MI = Σ over symbol pairs (x, y) of p(x, y) · log( p(x, y) / (p(x) · p(y)) )

where the probabilities are frequencies in the data, and pairs that never occur contribute nothing.

\`\`\`text
mutual_information("AABB", "CCDD")   ->  0.693147    (log 2: each column fully predicts the other)
mutual_information("AABB", "CDCD")   ->  0.0
\`\`\``,
        starter: `
import math
from collections import Counter

def mutual_information(a, b):
    # TODO
    pass
`,
        solution: `
import math
from collections import Counter

def mutual_information(a, b):
    n = len(a)
    pa, pb, pab = Counter(a), Counter(b), Counter(zip(a, b))
    mi = 0.0
    for (x, y), c in pab.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((pa[x] / n) * (pb[y] / n)))
    return mi
`,
        hints: ['Counter(zip(a, b)) counts the pairs.', 'Only pairs that occur need to be summed.'],
        explain: `Mutual information is zero when two columns are independent and grows as one predicts the other. On real alignments it is biased upwards for columns with many symbols and confused by indirect correlations (A with B, B with C makes A look related to C), which is why contact-prediction methods moved to corrected scores, global statistical models and finally to deep networks.`,
        tests: [
          { call: 'round(mutual_information("AABB", "CCDD"), 6)', expect: '0.693147' },
          { name: 'independent columns', call: 'round(mutual_information("AABB", "CDCD"), 6)', expect: '0.0' },
          { name: 'a constant column shares nothing', call: 'round(mutual_information("KKKK", "ACDE"), 6)', expect: '0.0' },
          { name: 'random columns vs reference', setup: 'import random\nrandom.seed(9)\nx = [random.choice("ACDE") for _ in range(60)]\ny = [c if random.random() < 0.7 else random.choice("ACDE") for c in x]', call: 'round(mutual_information(x, y), 6)', hidden: true }
        ]
      },
      {
        id: 'ensemble-errors', title: 'An Ensemble Forecast versus Its Members', difficulty: 'easy',
        prompt: `An ensemble forecast is a list of member forecasts (each a list of numbers, such as temperatures at several places). Implement \`ensemble_errors(members, truth)\` that returns a tuple:

- the **average error of the individual members**, where the error of a forecast is its Euclidean distance to \`truth\`;
- the **error of the ensemble mean** (the element-wise average of the members).

\`\`\`text
ensemble_errors([[1, 0], [-1, 0]], [0, 0])   ->  (1.0, 0.0)
\`\`\``,
        starter: `
import math

def ensemble_errors(members, truth):
    # TODO
    pass
`,
        solution: `
import math

def ensemble_errors(members, truth):
    dist = lambda f: math.sqrt(sum((x - t) ** 2 for x, t in zip(f, truth)))
    mean = [sum(col) / len(members) for col in zip(*members)]
    return sum(dist(m) for m in members) / len(members), dist(mean)
`,
        hints: ['zip(*members) walks through the members position by position.', 'Compute the member errors first, then the error of the averaged forecast.'],
        explain: `By the triangle inequality, the ensemble mean is never further from the truth than the average member, and usually much closer when members err in different directions, as they do far ahead in chaotic systems. The spread of the members is just as useful: it tells you how uncertain the forecast is.`,
        tests: [
          { call: 'ensemble_errors([[1, 0], [-1, 0]], [0, 0])', expect: '(1.0, 0.0)' },
          { name: 'identical members', call: 'ensemble_errors([[3, 4], [3, 4]], [0, 0])', expect: '(5.0, 5.0)' },
          { name: 'the mean never does worse', setup: 'import random\nrandom.seed(1)\nms = [[random.gauss(0, 1) for _ in range(5)] for _ in range(20)]', call: 'ensemble_errors(ms, [0] * 5)[1] <= ensemble_errors(ms, [0] * 5)[0]', expect: 'True' },
          { name: 'random members vs reference', setup: 'import random\nrandom.seed(6)\nms = [[random.gauss(1, 2) for _ in range(4)] for _ in range(10)]', call: '[round(x, 6) for x in ensemble_errors(ms, [0.5, 1.0, 1.5, 2.0])]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Jumper et al., Highly accurate protein structure prediction with AlphaFold (Nature, 2021)', url: 'https://www.nature.com/articles/s41586-021-03819-2' },
      { title: 'Lam et al., GraphCast: Learning skillful medium-range global weather forecasting (2022)', url: 'https://arxiv.org/abs/2212.12794' },
      { title: 'Merchant et al., Scaling deep learning for materials discovery (GNoME, Nature, 2023)', url: 'https://www.nature.com/articles/s41586-023-06735-9' }
    ]
  },

  // =====================================================================================
  {
    id: 'audio-generation', branch: 'spec', title: 'Audio & Music Generation', short: 'Audio Generation', level: 3, minutes: 45, prereqs: ['speech-audio', 'diffusion-models'],
    tags: ['audio generation', 'music generation', 'text-to-speech', 'tts', 'voice cloning', 'neural audio codec', 'residual vector quantization', 'rvq', 'encodec', 'soundstream', 'codec language model', 'delay pattern', 'musicgen', 'audio diffusion', 'fad'],
    summary: 'How models generate speech, sound and music: neural audio codecs that turn waveforms into a few tokens per second with residual vector quantization, language models over codec tokens and the delay pattern for generating several codebooks at once, diffusion and flow-matching audio models, text-to-speech and voice cloning, music generation and its copyright questions, and how generated audio is evaluated.',
    why: `Audio is the hardest everyday signal to generate: tens of thousands of samples per second, and listeners notice tiny glitches. The breakthrough was to stop generating samples directly. Neural codecs compress sound into short sequences of discrete tokens, and the same language models and diffusion models used for text and images then generate those tokens. The same pipeline powers voice assistants, audiobooks, dubbing, music tools and, unfortunately, voice-cloning fraud.`,
    objectives: [
      'Explain how a neural audio codec turns sound into tokens with residual vector quantization',
      'Compute token rates and bitrates, and explain the delay pattern for multi-codebook generation',
      'Describe how speech and music are generated with codec language models and diffusion models',
      'Evaluate generated audio and reason about voice-cloning and copyright risks'
    ],
    sections: [
      { type: 'text', title: "Sound as data", md: `Digital audio is a long list of air-pressure measurements: 16,000 per second for speech, 44,100 or 48,000 per second for music, per channel. A three-minute stereo song at CD quality is about 16 million numbers. Generating them directly is hard for three reasons:

- **Length.** Even a short clip is a sequence far longer than any text a language model writes.
- **Many time scales at once.** A phoneme lasts about a tenth of a second, a melodic phrase several seconds, a song's structure minutes.
- **Unforgiving listeners.** A click, a metallic tone or a slightly wrong rhythm is immediately audible.

WaveNet (2016) generated speech one sample at a time with very natural results, but its first version was far too slow for real use. Spectrograms (see [Speech & Audio](#/concept/speech-audio)) compress time but need a separate vocoder to turn them back into sound. The approach that took over is to compress audio into a short sequence of **discrete tokens** with a learned codec, and then generate the tokens.` },
      { type: 'text', title: "Neural audio codecs: sound as tokens", md: `A **neural audio codec** has three parts:

1. an **encoder**, a convolutional network that turns the waveform into a sequence of vectors, typically 25 to 75 frames per second;
2. a **quantizer** that replaces each vector with a few discrete codes using **residual vector quantization** (RVQ): the first codebook picks the code closest to the vector, the second picks the code closest to what is left over (the residual), and so on;
3. a **decoder** that turns the codes back into a waveform.

The codec is trained to reconstruct its input, with extra *adversarial* losses (discriminator networks that judge whether audio sounds real), so that the output sounds natural even where it is not sample-for-sample identical. SoundStream (2021), EnCodec (2022) and the Descript Audio Codec are well-known examples. The size of the token stream is easy to compute:

$$ bitrate = frame rate · codebooks · log₂(codebook size) $$
:::where
frame rate — encoder frames per second, typically 25 to 75
codebooks — how many residual codebooks quantize each frame
codebook size — the number of codes in each codebook; log₂ of it is the number of bits per code (10 bits for 1,024 codes)
bitrate — bits per second of the token stream
:::

With 50 frames per second, 4 codebooks and 1,024 codes, that is 2 kilobits per second, about 700 times less than CD audio.` },
      { type: 'code', title: "Residual vector quantization, step by step", md: `Stand-in encoder outputs (8 numbers per frame) quantized by up to eight codebooks of 16 codes each, every codebook learned with k-means on what the previous ones left over:`, code: `import numpy as np
rng = np.random.default_rng(0)
D, K = 8, 16                                           # numbers per frame, codes per codebook (real codecs: about 1,024)
frames = rng.normal(size=(4000, D)) @ rng.normal(size=(D, D)) * 0.5   # stand-in for an encoder's output, one row per frame

def kmeans(x, k, iters=15):
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        label = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([x[label == j].mean(0) if np.any(label == j) else c[j] for j in range(k)])
    return c

residual, errors = frames.copy(), []
print("codebooks   bits per frame   remaining error")
for level in range(1, 9):
    codebook = kmeans(residual, K)                     # each codebook quantizes what the previous ones missed
    index = ((residual[:, None, :] - codebook[None]) ** 2).sum(-1).argmin(1)
    residual = residual - codebook[index]
    errors.append(np.linalg.norm(residual) / np.linalg.norm(frames))
    print(f"{level:9d} {level * int(np.log2(K)):16d} {errors[-1]:17.3f}")
print("every extra codebook lowers the error:", all(a > b for a, b in zip(errors, errors[1:])))`, after: `One codebook of 16 codes leaves 62% of the signal unexplained, because 16 codes cannot cover an 8-dimensional space finely. Each further codebook only has to describe the previous stages' mistakes, and after eight (32 bits per frame) the error is down to under 8%, with smaller gains at each step. Eight codebooks of 16 codes can address 16⁸ (about 4 billion) combinations, yet each one is tiny to store and to search. Dropping the later codebooks lowers quality gradually, which lets one codec serve several bitrates.`, predict: {"type":"mcq","q":"Each frame is quantized by up to eight codebooks, each describing what the previous ones missed. What happens to the remaining error as codebooks are added?","options":["It stays the same after the first codebook","It falls with every codebook, by smaller amounts each time","It drops to zero after two codebooks","It rises, because the errors add up"],"answer":1,"explain":"Every codebook removes part of the remaining residual, so the error keeps falling (from 0.62 to 0.08 here), with diminishing returns.","expect":["every extra codebook lowers the error: True"]} },
      { type: 'chart', title: "Remaining error against bits per frame", md: `The stages from the code above, and the same procedure with codebooks of 64 codes:`, after: `Each codebook removes a similar share of what is left, about a quarter, so the error falls quickly at first and then more slowly. At the same number of bits, codebooks of 64 codes leave less error than codebooks of 16 (0.10 against 0.13 at 24 bits) and need fewer stages, so a language model has fewer tokens to generate per frame. Real codecs use about 1,024 codes per codebook for the same reasons; the price is a larger codebook to learn and search.`, chart: {"kind":"line","series":[{"name":"codebooks of 16 codes (4 bits each)","x":[4,8,12,16,20,24,28,32],"y":[0.615,0.4259,0.3074,0.2264,0.1708,0.1295,0.0988,0.07609]},{"name":"codebooks of 64 codes (6 bits each)","x":[6,12,18,24,30,36,42,48],"y":[0.4848,0.279,0.1687,0.1041,0.06505,0.04101,0.02628,0.01717]}],"xLabel":"bits per frame (one point per codebook added)","yLabel":"remaining error (share of the signal)","yMin":0} },
      { type: 'text', title: "Generating the tokens", md: `- **Codec language models** treat audio tokens like text. AudioLM (2022) continued speech and piano recordings convincingly, VALL-E (2023) recast text-to-speech as language modelling over codec tokens and could imitate a voice from a three-second sample, and MusicGen (2023) generates music tokens from a text description.
- **Several codebooks per frame** multiply the length of the sequence if tokens are generated one at a time. MusicGen's **delay pattern** instead predicts all codebooks in parallel, each shifted one step behind the previous one, so a clip of T frames takes about T + K − 1 steps for K codebooks.
- **Diffusion and flow matching** generate continuous latents or spectrograms instead of tokens: Stable Audio uses latent diffusion conditioned on text and timing, and flow-matching models such as Voicebox (2023) fill in and edit speech.
- **Speech-to-speech models** listen and answer in audio directly, which removes the delay of transcribing and synthesizing separately (see [Speech & Audio](#/concept/speech-audio)).` },
      { type: 'code', title: "How many tokens is a song?", code: `seconds = 180                                          # a three-minute song
print("codec                              bitrate    tokens for 3 minutes   steps, one token at a time   steps with a delay pattern")
for name, rate, books, bits in [("speech codec, 75 frames/s, 8 books", 75, 8, 10), ("music codec, 50 frames/s, 4 books", 50, 4, 10)]:
    tokens = seconds * rate * books
    print(f"{name:34s} {rate * books * bits / 1000:5.1f} kb/s {tokens:16,} {tokens:24,} {seconds * rate + books - 1:26,}")
cd = 44_100 * 16 * 2 / 1000
print(f"CD audio for comparison: {cd:,.0f} kb/s, about {cd / 2.0:,.0f} times the music codec's bitrate")
steps_delay = seconds * 50 + 4 - 1
print("the delay pattern needs about a quarter of the steps:", 0.24 < steps_delay / (seconds * 50 * 4) < 0.26)`, after: `Even at 2 kb/s, a three-minute song is 36,000 tokens, longer than most documents a language model reads. Generating one token at a time would take 36,000 sequential steps; the delay pattern needs 9,003, because the four codebooks of a frame are predicted together. Long pieces are still produced in overlapping windows, or with a model that plans coarse structure first and fills in detail afterwards.`, predict: {"type":"mcq","q":"A music codec produces 50 frames per second with 4 codebooks. With the delay pattern, roughly how many generation steps does a three-minute song need?","options":["About 180","About 9,000","About 36,000","About 1.4 million"],"answer":1,"explain":"180 s × 50 frames = 9,000 frames, plus 3 extra steps for the staggered codebooks: 9,003. Generating every token separately would take 36,000.","expect":["the delay pattern needs about a quarter of the steps: True"]} },
      { type: 'text', title: "Text-to-speech and voice cloning", md: `A text-to-speech system normalizes the text ("Dr." becomes "doctor", "1999" becomes "nineteen ninety-nine"), converts it into tokens or phonemes, generates acoustic tokens or a spectrogram, and decodes them into a waveform. The hardest part is **prosody**, the rhythm, stress and intonation that make speech sound natural, which depends on meaning and context.

**Voice cloning** from a few seconds of audio is now routine. It restores voices to people who have lost theirs to illness, and makes dubbing and audiobooks cheaper. It also enables fraud, such as phone calls in which a cloned voice of a relative or an executive asks for money. Responsible systems require the speaker's explicit consent and verify it, refuse to clone public figures, add inaudible **watermarks** to generated audio (Meta's AudioSeal, 2024, is one open example), and support detection tools, which are useful but imperfect.` },
      { type: 'text', title: "Music generation and its open questions", md: `- **Symbolic or audio.** Some models write notes (MIDI), which a musician or synthesizer then performs; most recent systems generate the sound itself, including vocals.
- **Control.** Text descriptions, a hummed melody, chords, tempo or the structure of a song (verse, chorus) can condition the output.
- **Evaluation.** The Fréchet audio distance (FAD) compares embeddings of real and generated audio, like FID for images; text–audio similarity models (CLAP) score how well a clip matches its prompt. Listening tests remain essential, because these metrics miss much of what listeners hear.
- **Copyright.** Many music models were trained on commercial recordings. In 2024 the major record labels sued two prominent AI music companies over their training data, and by late 2025 some of these disputes were turning into licensing agreements. Generated pieces can also come uncomfortably close to existing songs, so commercial use needs care (see [AI Governance & Regulation](#/concept/ai-governance)).` },
      { type: 'check', question: {"type":"tf","q":"A neural audio codec reproduces every sample of the original waveform exactly, like a ZIP file.","answer":false,"explain":"Neural codecs are lossy: they are trained to sound right to listeners, with adversarial losses, not to reproduce every sample."} }
    ],
    when: `- Text-to-speech for products, accessibility and dubbing, using a licensed or consented voice.
- Sound effects and music sketches for prototypes, games and video.
- Compressing audio for transmission or storage with neural codecs.`,
    pitfalls: `- Cloning a real person's voice without explicit consent.
- Ignoring the training data and licence terms of music models: copyright disputes over AI music are ongoing.
- Judging quality only by distribution metrics: listening tests remain essential.
- Underestimating sequence length: even compressed, a three-minute song is tens of thousands of tokens.`,
    takeaways: [
      'Neural codecs encode audio into a few dozen frames per second, each quantized by several residual codebooks.',
      'Each extra codebook in residual vector quantization refines what the previous ones missed.',
      'Codec language models generate audio tokens like text; the delay pattern generates several codebooks in parallel.',
      'Diffusion and flow-matching models generate audio latents or spectrograms, often conditioned on text and timing.',
      'Voice cloning needs consent and safeguards; music generation raises unresolved copyright questions.'
    ],
    quiz: [
      {"type":"fill","q":"A codec produces 75 frames per second with 8 codebooks of 1,024 codes. What is its bitrate in kilobits per second?","answer":["6","6.0"],"explain":"75 × 8 × 10 bits = 6,000 bits per second = 6 kb/s."},
      {"type":"mcq","q":"In residual vector quantization, what does the second codebook encode?","options":["A copy of the first code","The error left over after the first codebook's approximation","The next frame","The loudness only"],"answer":1,"explain":"Each stage quantizes the residual of the stages before it, so the approximation improves step by step."},
      {"type":"mcq","q":"What does MusicGen's delay pattern achieve?","options":["It removes codebooks","It predicts all codebooks of a frame in parallel, staggered by one step, so generation takes about frames + codebooks − 1 steps","It slows generation down to improve quality","It converts audio to MIDI"],"answer":1,"explain":"Staggering the codebooks lets one transformer step emit one code per codebook instead of generating them one at a time."},
      {"type":"tf","q":"Codec language models such as VALL-E can imitate a voice from a few seconds of example audio.","answer":true,"explain":"Which is why consent, verification and watermarking matter for any product that offers voice cloning."},
      {"type":"mcq","q":"What does the Fréchet audio distance measure?","options":["The length of a clip","The distance between the distributions of embeddings of real and generated audio","How loud a clip is","How many tokens a clip has"],"answer":1,"explain":"Like FID for images, it fits Gaussians to embeddings of real and generated clips and measures how far apart they are; lower is better."},
      {"type":"mcq","q":"Which practice belongs in a responsible voice-cloning product?","options":["Cloning any voice found online","Verified consent from the speaker, plus watermarks on generated audio","Hiding that audio is generated","Removing all limits on use"],"answer":1,"explain":"Consent addresses the person whose voice is used; watermarks and detection help others recognize generated audio."}
    ],
    problems: [
      {
        id: 'codec-bitrate', title: 'Bitrate of a Neural Codec', difficulty: 'easy',
        prompt: `A neural audio codec produces \`frame_rate\` frames per second, and quantizes each frame with \`codebooks\` codebooks of \`codebook_size\` entries each. Every code is an index that takes log₂(codebook_size) bits. Implement \`bitrate_kbps(frame_rate, codebooks, codebook_size)\` returning the bitrate in kilobits per second.

\`\`\`text
bitrate_kbps(50, 4, 1024)   ->  2.0      50 frames × 4 codes × 10 bits = 2,000 bits per second
\`\`\``,
        starter: `
import math

def bitrate_kbps(frame_rate, codebooks, codebook_size):
    # TODO
    pass
`,
        solution: `
import math

def bitrate_kbps(frame_rate, codebooks, codebook_size):
    return frame_rate * codebooks * math.log2(codebook_size) / 1000
`,
        hints: ['A codebook of 1,024 entries needs log₂(1024) = 10 bits per code.'],
        explain: `Uncompressed CD audio runs at about 1,411 kb/s; neural codecs reach usable quality at a few kb/s. Dropping codebooks lowers the bitrate smoothly, which is how one codec can serve several qualities.`,
        tests: [
          { call: 'bitrate_kbps(50, 4, 1024)', expect: '2.0' },
          { name: 'more codebooks', call: 'bitrate_kbps(75, 32, 1024)', expect: '24.0' },
          { name: 'smaller codebooks', call: 'bitrate_kbps(75, 8, 256)', expect: '4.8' },
          { name: 'another codec', call: 'bitrate_kbps(25, 12, 2048)', hidden: true }
        ]
      },
      {
        id: 'rvq-encode', title: 'Residual Vector Quantization', difficulty: 'medium',
        prompt: `Implement \`rvq_encode(x, codebooks)\`. \`x\` is a vector (a list of numbers) and \`codebooks\` is a list of codebooks, each a list of code vectors of the same length as x. For each codebook in turn:

1. pick the code vector closest to the **current residual** (smallest squared distance; on a tie, the lowest index);
2. record its index;
3. subtract it from the residual.

The residual starts as x. Return the list of indices, one per codebook.

\`\`\`text
rvq_encode([1.2, 0.0], [[[0, 0], [1, 0]], [[0.25, 0], [-0.25, 0]]])   ->  [1, 0]
\`\`\``,
        starter: `
def rvq_encode(x, codebooks):
    # TODO
    pass
`,
        solution: `
def rvq_encode(x, codebooks):
    residual, indices = list(x), []
    for book in codebooks:
        dists = [sum((r - c) ** 2 for r, c in zip(residual, code)) for code in book]
        k = dists.index(min(dists))
        indices.append(k)
        residual = [r - c for r, c in zip(residual, book[k])]
    return indices
`,
        hints: ['list.index(min(dists)) returns the first index of the smallest distance.', 'Update the residual after each codebook, before moving to the next one.'],
        explain: `Each stage only has to describe what the previous stages got wrong, so a few small codebooks together act like one enormous codebook (8 codebooks of 1,024 entries can address 1,024⁸ combinations), while staying cheap to search and to learn.`,
        tests: [
          { call: 'rvq_encode([1.2, 0.0], [[[0, 0], [1, 0]], [[0.25, 0], [-0.25, 0]]])', expect: '[1, 0]' },
          { name: 'a second stage that corrects the first', call: 'rvq_encode([0.0, 0.9], [[[0, 1], [1, 0]], [[0, 0.1], [0, -0.1]]])', expect: '[0, 1]' },
          { name: 'ties go to the lowest index', call: 'rvq_encode([0.5], [[[0], [1]]])', expect: '[0]' },
          { name: 'random codebooks vs reference', setup: 'import random\nrandom.seed(12)\nbooks = [[[random.gauss(0, 1) for _ in range(3)] for _ in range(8)] for _ in range(4)]\nv = [random.gauss(0, 1) for _ in range(3)]', call: 'rvq_encode(v, books)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Zeghidour et al., SoundStream: An End-to-End Neural Audio Codec (2021)', url: 'https://arxiv.org/abs/2107.03312' },
      { title: 'Défossez et al., High Fidelity Neural Audio Compression (EnCodec, 2022)', url: 'https://arxiv.org/abs/2210.13438' },
      { title: 'Copet et al., Simple and Controllable Music Generation (MusicGen, 2023)', url: 'https://arxiv.org/abs/2306.05284' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
