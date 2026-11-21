/* ML Atlas – content: Specializations (RL, generative models, vision transformers, recommenders, MLOps, responsible AI) */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'reinforcement-learning', branch: 'spec', title: 'Reinforcement Learning', short: 'Reinforcement Learning', level: 2, minutes: 50, prereqs: ['probability', 'supervised-learning'],
    tags: ['reinforcement learning', 'agent', 'environment', 'reward', 'policy', 'bandits', 'exploration', 'value function', 'bellman equation', 'q-learning', 'policy gradient', 'ppo'],
    summary: 'Learning by trial and error: agents, environments, states, actions and rewards; multi-armed bandits and the exploration–exploitation trade-off; discounted returns, value functions and the Bellman equation; Q-learning; and policy gradients (the idea behind RLHF).',
    why: `Supervised learning needs the correct answer for every example. For many problems nobody can provide that (the best move in a game, the right motor command for a robot, the ideal reply of a chatbot) but you *can* score the outcome. **Reinforcement learning** learns from such scores by trial and error. It produced AlphaGo and robot locomotion, optimizes data-centre cooling and chip layouts, and is the "RL" in RLHF and in the training of reasoning models.`,
    objectives: [
      'Describe the RL loop: agent, environment, state, action, reward and policy',
      'Balance exploration and exploitation in a multi-armed bandit with ε-greedy',
      'Compute discounted returns and explain value functions and the Bellman equation',
      'Implement the Q-learning update and explain policy gradients at a high level'
    ],
    sections: [
      { type: 'text', title: 'The reinforcement-learning loop', md: `An **agent** interacts with an **environment** in steps. At each step it observes a **state**, chooses an **action** according to its **policy**, and receives a **reward** and the next state:

\`\`\`text
          action a_t
   agent ------------> environment
     ^                     |
     +---------------------+
       reward r_t+1, next state s_t+1
\`\`\`

The goal is to maximize the **total reward over time**, not the immediate reward, sometimes you must sacrifice now to win later.

| Problem | State | Action | Reward |
| --- | --- | --- | --- |
| Chess or Go | board position | a move | +1 win, −1 loss, at the end |
| Robot walking | joint angles, speeds | motor torques | forward progress, minus energy |
| Recommendations | user context | which item to show | click or watch time |
| LLM fine-tuning (RLHF) | prompt + text so far | the next token | a preference score for the finished reply |

Compared with supervised learning: there are no labels, only rewards, often **delayed** (which move actually lost the game?, the credit-assignment problem); the agent's own actions decide which data it sees; and it must **explore** to discover what works.` },
      { type: 'text', title: 'Multi-armed bandits: explore or exploit?', md: `The simplest RL problem: K slot machines ("arms") with unknown average payouts. Each turn you pull one arm and receive a random reward. Estimate each arm's value as the average reward you have seen from it, updated incrementally:

$$ Q(a) ← Q(a) + ( r − Q(a) ) / N(a) $$
:::where
a — one of the arms
Q(a) — the current estimate of arm a's average reward
r — the reward just received from arm a
N(a) — how many times arm a has been pulled, including this pull
( r − Q(a) ) / N(a) — move the estimate a fraction of the way towards the new reward, which keeps it equal to the running average
:::

Always pulling the best-looking arm (**exploitation**) can lock onto a mediocre arm forever; trying others (**exploration**) costs reward now but finds better arms. **ε-greedy** balances the two: with probability ε pick a random arm, otherwise the arm with the highest Q. Smarter strategies include UCB (optimism about rarely tried arms) and Thompson sampling. Bandits are used for A/B testing, ad selection and recommendations.` },
      { type: 'code', title: 'ε-greedy on a 5-armed bandit', code: `
import numpy as np
true_means = np.array([0.2, 0.5, 0.9, 0.4, 0.1])     # unknown to the agent; arm 2 is best

def run(epsilon, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    q, n, total = np.zeros(5), np.zeros(5), 0.0
    for t in range(steps):
        arm = int(rng.integers(5)) if rng.random() < epsilon else int(np.argmax(q))   # explore or exploit
        reward = rng.normal(true_means[arm], 1.0)                                      # noisy payout
        n[arm] += 1
        q[arm] += (reward - q[arm]) / n[arm]                                           # incremental average
        total += reward
    return total / steps, int(np.argmax(q))

for eps in (0.0, 0.01, 0.1, 0.3):
    results = [run(eps, seed=s) for s in range(20)]
    avg = np.mean([r for r, _ in results])
    found = sum(best == 2 for _, best in results)
    print(f"epsilon={eps:<5} average reward {avg:.3f}   identified the best arm in {found}/20 runs")
`, after: `Pure greed (ε = 0) often gets stuck on whichever arm looked good first. A little exploration (ε = 0.1) earns the most; too much (ε = 0.3) finds the best arm every time but wastes pulls on bad arms.` },
      { type: 'chart', title: "How often each strategy picks the best arm", md: `The 5-armed bandit above (rewards are noisy: standard deviation 1), averaged over 2,000 independent runs:`, after: `The purely greedy agent (ε = 0) locks onto whichever arm looked good early and finds the best arm in only about 38% of runs. Exploring 10% of the time finds it far more reliably and soon picks it about 90% of the time, the most it can while still exploring 10% of the time. ε = 0.01 learns slowly but would end up higher in the long run, and ε = 0.3 explores too much. Annealing ε from high to low, or exploring by uncertainty (UCB, Thompson sampling), gets the best of both.`, chart: {"kind":"line","x":[1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,321,331,341,351,361,371,381,391,401,411,421,431,441,451,461,471,481,491,501,511,521,531,541,551,561,571,581,591,601,611,621,631,641,651,661,671,681,691,701,711,721,731,741,751,761,771,781,791,801,811,821,831,841,851,861,871,881,891,901,911,921,931,941,951,961,971,981,991],"points":false,"series":[{"name":"ε = 0.0","y":[0.278,0.351,0.363,0.367,0.371,0.374,0.375,0.376,0.377,0.378,0.379,0.379,0.38,0.38,0.381,0.381,0.381,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382,0.382]},{"name":"ε = 0.01","y":[0.303,0.371,0.389,0.4,0.403,0.409,0.418,0.424,0.431,0.437,0.443,0.45,0.455,0.462,0.466,0.472,0.476,0.484,0.488,0.492,0.498,0.499,0.504,0.507,0.51,0.516,0.517,0.521,0.526,0.529,0.53,0.535,0.539,0.543,0.545,0.55,0.555,0.556,0.56,0.563,0.568,0.569,0.574,0.579,0.584,0.587,0.589,0.592,0.597,0.601,0.604,0.608,0.61,0.613,0.615,0.616,0.62,0.619,0.621,0.623,0.628,0.629,0.631,0.635,0.635,0.638,0.639,0.642,0.645,0.647,0.653,0.657,0.658,0.663,0.665,0.67,0.672,0.673,0.677,0.679,0.679,0.682,0.684,0.683,0.686,0.687,0.69,0.692,0.693,0.695,0.695,0.698,0.7,0.701,0.704,0.706,0.706,0.708,0.711,0.712]},{"name":"ε = 0.1","y":[0.283,0.366,0.416,0.464,0.505,0.532,0.564,0.578,0.603,0.632,0.644,0.656,0.677,0.693,0.708,0.721,0.732,0.742,0.756,0.763,0.771,0.784,0.786,0.792,0.793,0.803,0.805,0.813,0.817,0.822,0.827,0.826,0.831,0.84,0.841,0.846,0.841,0.849,0.85,0.852,0.857,0.858,0.859,0.861,0.863,0.866,0.866,0.87,0.87,0.871,0.876,0.875,0.879,0.876,0.88,0.879,0.884,0.88,0.886,0.883,0.888,0.889,0.889,0.888,0.886,0.891,0.891,0.891,0.892,0.892,0.894,0.895,0.897,0.893,0.895,0.893,0.894,0.898,0.894,0.896,0.898,0.897,0.9,0.899,0.896,0.895,0.899,0.898,0.899,0.902,0.898,0.903,0.903,0.904,0.903,0.902,0.901,0.903,0.903,0.906]},{"name":"ε = 0.3","y":[0.277,0.372,0.436,0.49,0.532,0.572,0.597,0.613,0.633,0.645,0.653,0.665,0.679,0.686,0.692,0.7,0.699,0.711,0.72,0.715,0.724,0.724,0.728,0.731,0.733,0.735,0.74,0.741,0.744,0.739,0.741,0.749,0.747,0.75,0.752,0.753,0.754,0.754,0.757,0.758,0.755,0.754,0.755,0.756,0.757,0.749,0.755,0.752,0.749,0.758,0.758,0.758,0.753,0.757,0.761,0.756,0.756,0.76,0.754,0.756,0.755,0.754,0.759,0.759,0.757,0.754,0.759,0.761,0.76,0.759,0.761,0.758,0.756,0.756,0.759,0.759,0.762,0.763,0.759,0.761,0.759,0.763,0.761,0.76,0.767,0.759,0.762,0.76,0.76,0.763,0.761,0.759,0.759,0.762,0.76,0.766,0.757,0.757,0.76,0.761]}],"xLabel":"step","yLabel":"share of 2,000 runs choosing the best arm","yMin":0,"yMax":1} },
      { type: 'check', question: { type: 'mcq', q: 'In ε-greedy with ε = 0.1, what happens on a typical step?', options: ['Always a random arm', 'With probability 0.9 the best-looking arm, with probability 0.1 a random arm', 'The arm pulled least often', 'Arms in a fixed rotation'], answer: 1, explain: 'Mostly exploit, occasionally explore.' } },
      { type: 'text', title: 'Returns and discounting', md: `In longer tasks, actions affect future rewards. The **return** from time t is the sum of all future rewards, with later rewards discounted:

$$ G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + … = Σ_k γ^k r_{t+k+1} $$
:::where
G_t — the return: everything the agent will collect from time t onwards
r_{t+1} — the reward received right after the action at time t (and so on for later steps)
γ — "gamma": the discount factor between 0 and 1, so a reward k steps away counts γ^k as much
γ^k — γ raised to the power k
Σ_k — add up over all future steps k = 0, 1, 2, …
:::

Discounting makes the agent prefer sooner rewards and keeps infinite sums finite. Roughly, the agent plans about 1 / (1 − γ) steps ahead: γ = 0.99 means about 100 steps.` },
      { type: 'text', title: 'Value functions and the Bellman equation', md: `A **value function** predicts the expected return. Q(s, a) (the **action value**) is the expected return after taking action a in state s and acting well afterwards. If you knew the optimal Q*, acting optimally would be easy: pick the action with the highest Q* in every state.

Q* satisfies a recursive consistency condition, the **Bellman optimality equation**:

$$ Q*(s, a) = E[ r + γ · max_a′ Q*(s′, a′) ] $$
:::where
s, a — the current state and the action taken
r — the immediate reward
s′ — the next state the environment moves to ("s prime")
max_a′ Q*(s′, a′) — the value of the best action available in the next state
E[ … ] — the expected (average) value over the environment's randomness
γ — the discount factor
:::

In words: the value of an action is the reward you get now plus the discounted value of the best you can do from where you land.` },
      { type: 'text', title: 'Q-learning', md: `**Q-learning** learns Q* from experience by nudging a table of estimates towards the Bellman target after every step (s, a, r, s′):

$$ Q(s, a) ← Q(s, a) + α [ r + γ · max_a′ Q(s′, a′) − Q(s, a) ] $$
:::where
Q(s, a) — the current estimate for taking action a in state s
α — "alpha": the learning rate, between 0 and 1
r + γ · max_a′ Q(s′, a′) — the target: the reward plus the discounted value of the best next action (just r if s′ ends the episode)
[ … ] — the temporal-difference (TD) error: how surprised the estimate was
:::

Q-learning is **off-policy**: it learns the values of the greedy policy while behaving ε-greedily to keep exploring. **Deep Q-Networks** (DQN, 2015) replaced the table with a neural network and learned to play dozens of Atari games from raw pixels.` },
      { type: 'code', title: 'Q-learning in a gridworld', code: `
import numpy as np
rng = np.random.default_rng(0)
H, W = 4, 4
GOAL, PIT = (3, 3), (1, 2)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
ARROWS = "^v<>"

def step(state, a):
    r, c = state[0] + ACTIONS[a][0], state[1] + ACTIONS[a][1]
    nxt = (min(max(r, 0), H - 1), min(max(c, 0), W - 1))    # walls keep the agent inside the grid
    if nxt == GOAL:
        return nxt, 1.0, True
    if nxt == PIT:
        return nxt, -1.0, True
    return nxt, -0.01, False                                  # a small cost per step rewards short paths

Q = np.zeros((H, W, 4))
alpha, gamma, eps = 0.5, 0.95, 0.1
for episode in range(500):
    s, done = (0, 0), False
    while not done:
        a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        target = r if done else r + gamma * Q[s2].max()
        Q[s][a] += alpha * (target - Q[s][a])                 # the Q-learning update
        s = s2

print("greedy policy (G = goal, X = pit):")
for r in range(H):
    print("  " + " ".join("G" if (r, c) == GOAL else "X" if (r, c) == PIT else ARROWS[int(np.argmax(Q[r, c]))] for c in range(W)))
print("estimated value of the start state:", round(float(Q[0, 0].max()), 3))
`, after: `Follow the arrows from the top-left corner: the agent learned a short route to the goal that steers around the pit. Cells it rarely visited can still show arbitrary arrows: it only learns about states it explores.` },
      { type: 'text', title: 'Policy gradients and modern RL', md: `Instead of learning values, **policy-gradient** methods adjust a policy network's parameters θ directly, making actions that led to high returns more likely:

$$ ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · A_t ] $$
:::where
θ — the parameters (weights) of the policy network
π_θ(a | s) — the policy: the probability of taking action a in state s
J(θ) — the expected return when acting with this policy
∇_θ log π_θ(a_t | s_t) — the direction in parameter space that makes the chosen action more likely
A_t — the advantage: how much better the action turned out than expected (the return minus a baseline)
:::

REINFORCE uses this directly; **actor–critic** methods learn a value function as the baseline; **PPO** clips each update so the policy can't change too much at once, which makes training stable: it is the algorithm behind classic RLHF. For an LLM the recipe reads: sample a reply, score it, and raise the log-probability of the tokens in good replies while lowering it in bad ones.` },
      { type: 'check', question: { type: 'tf', q: 'In Q-learning, the update moves Q(s, a) towards the reward plus the discounted value of the best action in the next state.', answer: true, explain: 'That target comes from the Bellman optimality equation.' } }
    ],
    when: `- Sequential decisions with delayed rewards where you can simulate or interact cheaply: games, robotics in simulation, control, scheduling, resource allocation.
- Bandits for online choices with immediate feedback: A/B tests, ads, recommendations.
- Fine-tuning LLMs from preference or verifiable rewards (RLHF, reasoning).
- Avoid it when labeled data exists (supervised learning is far more sample-efficient) or when exploring in the real world is dangerous or expensive.`,
    pitfalls: `- Reward hacking: the agent maximizes a badly specified reward in unintended ways.
- Too little exploration, so the agent settles for a mediocre strategy.
- Instability and strong sensitivity to hyperparameters and random seeds: compare over several seeds.
- Sample inefficiency: many problems need millions of episodes.
- The sim-to-real gap: policies trained in simulation can fail on real hardware.`,
    takeaways: [
      'RL learns a policy from rewards by trial and error, maximizing long-term return.',
      'Exploration vs exploitation: ε-greedy explores with probability ε.',
      'Return G = Σ γ^k r: discounting makes nearer rewards count more.',
      'Q-learning: Q(s, a) ← Q(s, a) + α[r + γ max Q(s′, ·) − Q(s, a)].',
      'Policy gradients (REINFORCE, PPO) push up the probability of advantageous actions: the basis of RLHF.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the main difference between reinforcement learning and supervised learning?', options: ['RL uses neural networks', 'RL learns from rewards for its own actions instead of from labelled correct answers', 'RL needs more labelled data', 'RL cannot handle images'], answer: 1, explain: 'The agent must discover good actions by trying them.' },
      { type: 'fill', q: 'Rewards 0, 0, 1 arrive at steps t+1, t+2, t+3 and γ = 0.9. What is the return G_t? (2 decimals)', answer: ['0.81'], explain: '0 + 0.9·0 + 0.9²·1 = 0.81.' },
      { type: 'mcq', q: 'Why is some exploration necessary?', options: ['It makes training faster', 'Without it the agent may never discover that other actions are better than its current favourite', 'It reduces the reward', 'Exploration is not necessary'], answer: 1, explain: 'Greedy agents can get stuck with early, misleading estimates.' },
      { type: 'mcq', q: 'In Q-learning, what is the target for Q(s, a) when s′ ends the episode?', options: ['0', 'Just the reward r', 'r + γ · max Q(s′, ·)', 'The average of Q'], answer: 1, explain: 'There is no future value after a terminal state.' },
      { type: 'mcq', q: 'Which RL algorithm is classically used in RLHF for language models?', options: ['k-means', 'PPO (a policy-gradient method)', 'Naive Bayes', 'Q-learning with a lookup table'], answer: 1, explain: 'PPO keeps policy updates small and stable.' },
      {"type":"mcq","q":"Why does a purely greedy bandit agent (ε = 0) often settle on a worse arm?","options":["It explores too much","It stops trying arms whose early estimates looked bad, so it never discovers that they are better","It forgets its rewards","Its learning rate is too high"],"answer":1,"explain":"Noisy early rewards can make the best arm look poor; without exploration that first impression is never corrected."},
      {"type":"tf","q":"With ε = 0.1 and 5 arms, an agent can never choose the best arm more than about 92% of the time, even once it knows which arm is best.","answer":true,"explain":"It explores 10% of the time, choosing uniformly among the 5 arms, so it picks the best arm with probability at most 0.9 + 0.1 × 1/5 = 0.92."}
    ],
    problems: [
      {
        id: 'discounted-return', title: 'Discounted Returns', difficulty: 'easy',
        prompt: `Here \`rewards[t]\` is the reward received right after the action at step t. Implement \`discounted_returns(rewards, gamma)\` → a list with the return for every step:

$$ G_t = rewards[t] + γ · rewards[t+1] + γ² · rewards[t+2] + … $$
:::where
G_t — the return from step t to the end of the episode
γ — gamma, the discount factor
rewards[t] — the reward that follows the action at step t
:::

Compute it backwards in a single pass: G_(T−1) = rewards[T−1] and G_t = rewards[t] + γ · G_(t+1).

\`\`\`text
discounted_returns([1, 1, 1], 0.5)  ->  [1.75, 1.5, 1.0]
discounted_returns([0, 0, 1], 0.9)  ->  [0.81, 0.9, 1.0]
\`\`\``,
        starter: `
def discounted_returns(rewards, gamma):
    # TODO: work backwards from the last reward
    pass
`,
        solution: `
def discounted_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns
`,
        hints: ['Loop with `range(len(rewards) - 1, -1, -1)`.', 'Keep a running total g and update it as g = r + gamma * g.'],
        explain: `Computing returns backwards turns a quadratic double loop into a single linear pass. Policy-gradient methods such as REINFORCE compute exactly these returns for every step of every episode.`,
        tests: [
          { call: 'discounted_returns([1, 1, 1], 0.5)', expect: '[1.75, 1.5, 1.0]' },
          { call: 'discounted_returns([0, 0, 1], 0.9)', expect: '[0.81, 0.9, 1.0]' },
          { name: 'gamma = 0 keeps only immediate rewards', call: 'discounted_returns([3, -1, 2], 0.0)', expect: '[3.0, -1.0, 2.0]', hidden: true },
          { name: 'gamma = 1 gives suffix sums', call: 'discounted_returns([1, 2, 3, 4], 1.0)', expect: '[10.0, 9.0, 7.0, 4.0]', hidden: true },
          { name: 'random episode', setup: 'rng = np.random.default_rng(0)\nrewards = rng.normal(size=25).tolist()', call: 'discounted_returns(rewards, 0.97)', hidden: true }
        ]
      },
      {
        id: 'epsilon-greedy-bandit', title: 'An ε-Greedy Bandit Agent', difficulty: 'medium',
        prompt: `Implement:

- \`choose_arm(q, epsilon, rng)\`: call \`rng.random()\` once; if it is \`< epsilon\` explore and return \`int(rng.integers(len(q)))\`, otherwise return \`int(np.argmax(q))\` (ties go to the lowest index).
- \`update(q, n, arm, reward)\` → new arrays \`(q2, n2)\` (don't modify the inputs): n2[arm] += 1, then q2[arm] += (reward − q2[arm]) / n2[arm].
- \`run_bandit(true_means, steps, epsilon, seed)\`: \`rng = np.random.default_rng(seed)\`, q and n start as zeros (floats). Each step: \`arm = choose_arm(q, epsilon, rng)\`, \`reward = rng.normal(true_means[arm], 1.0)\`, update q and n, and add the reward to a total. Return \`(q, n, total)\`.

Call the random generator exactly in this order so your results match the tests.`,
        starter: `
import numpy as np

def choose_arm(q, epsilon, rng):
    # TODO
    pass

def update(q, n, arm, reward):
    # TODO
    pass

def run_bandit(true_means, steps, epsilon, seed):
    # TODO
    pass
`,
        solution: `
import numpy as np

def choose_arm(q, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))

def update(q, n, arm, reward):
    q2, n2 = np.array(q, dtype=float), np.array(n, dtype=float)
    n2[arm] += 1
    q2[arm] += (reward - q2[arm]) / n2[arm]
    return q2, n2

def run_bandit(true_means, steps, epsilon, seed):
    rng = np.random.default_rng(seed)
    k = len(true_means)
    q, n, total = np.zeros(k), np.zeros(k), 0.0
    for _ in range(steps):
        arm = choose_arm(q, epsilon, rng)
        reward = rng.normal(true_means[arm], 1.0)
        q, n = update(q, n, arm, reward)
        total += reward
    return q, n, total
`,
        hints: ['`np.array(q, dtype=float)` makes a copy you can modify.', 'Only call `rng.integers` when exploring.', 'The incremental update keeps q equal to the running average of each arm\'s rewards.'],
        explain: `The incremental mean avoids storing every reward: each new reward moves the estimate by 1/N of the surprise. With ε > 0 every arm keeps getting sampled occasionally, so the estimates converge to the true means and the best arm is eventually exploited.`,
        tests: [
          { name: 'greedy choice', setup: 'rng = np.random.default_rng(0)', call: 'choose_arm(np.array([0.1, 0.7, 0.7, 0.2]), 0.0, rng)', expect: '1' },
          { call: 'update(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 1, 3.0)', expect: '(np.array([0.0, 2.0]), np.array([0.0, 2.0]))' },
          { name: 'a short run matches the reference', call: 'run_bandit([0.0, 1.0, 0.5], 50, 0.2, 7)' },
          { name: 'inputs are not modified', code: 'q, n = np.array([1.0, 2.0]), np.array([1.0, 1.0])\nupdate(q, n, 0, 5.0)\nassert q.tolist() == [1.0, 2.0] and n.tolist() == [1.0, 1.0]', hidden: true },
          { name: 'explores with epsilon = 1', setup: 'rng = np.random.default_rng(3)', call: '[choose_arm(np.zeros(5), 1.0, rng) for _ in range(8)]', hidden: true },
          { name: 'finds the best arm', code: 'q, n, total = run_bandit([0.2, 0.5, 0.9, 0.4], 2000, 0.1, 1)\nassert int(np.argmax(n)) == 2, n\nassert abs(q[2] - 0.9) < 0.15', hidden: true }
        ]
      },
      {
        id: 'q-learning-update', title: 'The Q-Learning Update', difficulty: 'medium',
        prompt: `Q is a NumPy array of shape (n_states, n_actions). Implement:

- \`q_update(Q, s, a, r, s_next, done, alpha, gamma)\` → a **new** array (don't modify Q) where only Q[s, a] changes:

$$ Q[s, a] ← Q[s, a] + α · ( target − Q[s, a] ),      target = r + γ · max(Q[s_next])   (or just r if done) $$
:::where
s, a — the state and the action that was taken
r — the reward received
s_next — the state the agent moved to
done — True when s_next ends the episode, so there is no future value
α, γ — the learning rate (alpha) and the discount factor (gamma)
max(Q[s_next]) — the value of the best action in the next state
:::

- \`greedy_policy(Q)\` → an integer array with the best action for every state (\`np.argmax\` along the actions axis).

\`\`\`text
Q = np.zeros((2, 2))
q_update(Q, 0, 1, 1.0, 1, False, 0.5, 0.9)  ->  [[0, 0.5], [0, 0]]
\`\`\``,
        starter: `
import numpy as np

def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    # TODO
    pass

def greedy_policy(Q):
    # TODO
    pass
`,
        solution: `
import numpy as np

def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    Q2 = np.array(Q, dtype=float)
    target = r if done else r + gamma * Q2[s_next].max()
    Q2[s, a] += alpha * (target - Q2[s, a])
    return Q2

def greedy_policy(Q):
    return np.argmax(np.asarray(Q), axis=1)
`,
        hints: ['Copy first with `np.array(Q, dtype=float)`.', 'Compute the target before changing anything.', '`np.argmax(Q, axis=1)` gives one action per state.'],
        explain: `Note that the target uses the *best* next action even if the agent will actually explore: that is what makes Q-learning off-policy. Applying this update over many episodes propagates the goal's reward backwards through the states, as in the gridworld example.`,
        tests: [
          { call: 'q_update(np.zeros((2, 2)), 0, 1, 1.0, 1, False, 0.5, 0.9)', expect: 'np.array([[0.0, 0.5], [0.0, 0.0]])' },
          { name: 'bootstraps from the next state', call: 'q_update(np.array([[0.0, 0.0], [2.0, 4.0]]), 0, 0, 1.0, 1, False, 0.5, 0.5)', expect: 'np.array([[1.5, 0.0], [2.0, 4.0]])' },
          { name: 'terminal transition ignores the future', call: 'q_update(np.array([[0.0, 0.0], [2.0, 4.0]]), 0, 0, 1.0, 1, True, 0.5, 0.5)', expect: 'np.array([[0.5, 0.0], [2.0, 4.0]])', hidden: true },
          { name: 'greedy policy', call: 'greedy_policy(np.array([[0.1, 0.9], [0.5, 0.2], [0.3, 0.3]]))', expect: 'np.array([1, 0, 0])', hidden: true },
          { name: 'a random sequence of transitions', setup: 'rng = np.random.default_rng(0)\nQ = np.zeros((5, 3))\nsteps = [(int(rng.integers(5)), int(rng.integers(3)), float(rng.normal()), int(rng.integers(5)), bool(rng.random() < 0.2)) for _ in range(40)]', code: 'Q_user = Q.copy()\nfor s, a, r, s2, d in steps:\n    Q_user = q_update(Q_user, s, a, r, s2, d, 0.3, 0.9)\nQ_ref = Q.copy()\nfor s, a, r, s2, d in steps:\n    target = r if d else r + 0.9 * Q_ref[s2].max()\n    Q_ref[s, a] += 0.3 * (target - Q_ref[s, a])\nassert np.allclose(Q_user, Q_ref)\nassert np.all(Q == 0), "the input must not be modified"', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Sutton & Barto, Reinforcement Learning: An Introduction (free book)', url: 'http://incompleteideas.net/book/the-book-2nd.html' },
      { title: 'Mnih et al., Human-level control through deep reinforcement learning (DQN)', url: 'https://www.nature.com/articles/nature14236' },
      { title: 'Schulman et al., Proximal Policy Optimization Algorithms', url: 'https://arxiv.org/abs/1707.06347' },
      { title: 'OpenAI, Spinning Up in Deep RL', url: 'https://spinningup.openai.com/' }
    ]
  },

  // =====================================================================================
  {
    id: 'generative-models', branch: 'spec', title: 'Generative Models: Autoencoders, VAEs, GANs & Diffusion', short: 'Generative Models', level: 3, minutes: 50, prereqs: ['mlp', 'probability'],
    tags: ['generative models', 'autoencoder', 'vae', 'latent space', 'reparameterization', 'gan', 'diffusion', 'denoising', 'stable diffusion'],
    summary: 'Models that create new data: autoencoders and latent spaces, variational autoencoders and the reparameterization trick, GANs (a generator against a discriminator), and diffusion models that learn to remove noise step by step (the technology behind modern image, audio and video generation).',
    why: `A classifier answers "what is this?"; a **generative model** learns what the data looks like well enough to produce new examples. Diffusion models power today's image and video generators; autoencoders compress data and learn useful representations; GANs produced the first photorealistic synthetic faces. LLMs are generative models too: autoregressive ones. Seeing the main families side by side shows the different ways to model a probability distribution over complex data.`,
    objectives: [
      'Explain what it means to model a data distribution and to sample from it',
      'Describe autoencoders and latent spaces; explain the VAE loss and the reparameterization trick',
      'Explain the GAN game between a generator and a discriminator, and its failure modes',
      'Explain the forward (noising) and reverse (denoising) processes of diffusion models'
    ],
    sections: [
      { type: 'text', title: 'What generative models do', md: `A generative model learns the probability distribution p(x) of its training data (images, sounds, sentences, molecules) so that it can **sample** new, realistic x. The main families take different routes:

| Family | How it generates | Examples | Trade-offs |
| --- | --- | --- | --- |
| Autoregressive | one piece at a time, each conditioned on the previous | GPT, PixelCNN, WaveNet | exact likelihoods; slow sequential sampling |
| Variational autoencoder (VAE) | decode a random latent vector | VAEs, the latent space of Stable Diffusion | fast, smooth latent space; blurrier samples |
| GAN | a generator network maps noise to data in one shot | StyleGAN | sharp and fast; unstable training |
| Diffusion | start from noise and denoise it step by step | Stable Diffusion, image and video models | top quality and diversity; many steps |` },
      { type: 'text', title: 'Autoencoders', md: `An **autoencoder** squeezes the input through a narrow bottleneck: an encoder maps x to a short **latent code** z, and a decoder tries to rebuild x from z. Training minimizes the reconstruction error:

$$ L = ‖ x − decoder(encoder(x)) ‖² $$
:::where
x — an input example, for instance an image flattened into a vector
encoder(x) — the latent code z: a much shorter vector summarizing x
decoder(z) — the reconstruction x̂ built from the code
‖ … ‖² — the squared distance between the input and its reconstruction
:::

Because z is small, the network must keep the most important structure and drop noise. Uses: compression, **denoising**, **anomaly detection** (unusual inputs reconstruct badly) and learning features. Plain autoencoders are poor generators, though: random points in their latent space often decode to garbage, because nothing makes the space smooth.` },
      { type: 'text', title: 'Variational autoencoders (VAEs)', md: `A **VAE** makes the latent space smooth and samplable. The encoder outputs a mean μ and a standard deviation σ for each latent dimension, a code z is *sampled* from that Gaussian, and the loss adds a penalty that keeps those Gaussians close to a standard normal:

$$ L = ‖ x − x̂ ‖² + KL( N(μ, σ²) ‖ N(0, 1) ) $$
$$ KL = ½ Σ_j ( μ_j² + σ_j² − 1 − log σ_j² ) $$
:::where
x̂ — the decoder's reconstruction of x from the sampled code z
μ_j, σ_j — the mean and standard deviation the encoder outputs for latent dimension j
N(μ, σ²) — a normal (Gaussian) distribution with mean μ and variance σ²
KL( … ‖ N(0, 1) ) — the KL divergence from the standard normal, which pulls every code towards a shared, smooth region
Σ_j — add up over the latent dimensions
:::

After training, sampling z from N(0, 1) and decoding it produces new data.

**The reparameterization trick.** "Sample z" is not differentiable, so gradients could not reach the encoder. Rewrite the sample as a deterministic function of the encoder's outputs plus external noise:

$$ z = μ + σ · ε,          ε ~ N(0, 1) $$
:::where
ε — "epsilon": standard normal noise, drawn independently of the network
~ — "is drawn from"
μ + σ · ε — has exactly the distribution N(μ, σ²), but now z changes smoothly with μ and σ, so gradients flow through them
:::` },
      { type: 'code', title: 'The reparameterization trick in action', code: `
import numpy as np
rng = np.random.default_rng(0)
mu, sigma = 1.5, 0.5
eps = rng.normal(size=100_000)
z = mu + sigma * eps                         # reparameterized samples of N(mu, sigma^2)
print(f"sample mean {z.mean():.3f} (mu = {mu})   sample std {z.std():.3f} (sigma = {sigma})")

# Gradients flow through the samples: the objective E[(z - 3)^2] = (mu - 3)^2 + sigma^2
# dz/dmu = 1 and dz/dsigma = eps, so we can estimate both gradients by averaging
print(f"d/dmu    estimated {np.mean(2 * (z - 3)):.3f}   exact {2 * (mu - 3):.3f}")
print(f"d/dsigma estimated {np.mean(2 * (z - 3) * eps):.3f}   exact {2 * sigma:.3f}")

def kl_to_standard_normal(mu, sigma):
    return 0.5 * (mu ** 2 + sigma ** 2 - 1 - np.log(sigma ** 2))
for m, s in [(0.0, 1.0), (1.5, 0.5), (0.0, 0.1)]:
    print(f"KL( N({m}, {s}^2) || N(0, 1) ) = {kl_to_standard_normal(m, s):.3f}")
`, after: `The Monte-Carlo gradient estimates match the exact values, which is how a VAE's encoder learns despite the random sampling. The KL penalty is 0 for a standard normal and grows when the code drifts away or becomes too narrow.` },
      { type: 'check', question: { type: 'mcq', q: 'Why do VAEs use the reparameterization trick?', options: ['To make sampling faster', 'To make the sampling step differentiable so gradients reach the encoder', 'To remove the KL term', 'To avoid using a decoder'], answer: 1, explain: 'Writing z = μ + σ·ε moves the randomness into ε.' } },
      { type: 'text', title: 'GANs: a forger and a detective', md: `A **generative adversarial network** trains two networks against each other. The **generator** G turns random noise z into a fake sample; the **discriminator** D outputs the probability that its input is real. D learns to tell real from fake; G learns to fool D:

$$ min_G max_D   E_x[ log D(x) ] + E_z[ log(1 − D(G(z))) ] $$
:::where
G(z) — a fake sample produced from random noise z
D(x) — the discriminator's probability that x is a real training example
E_x — the average over real data; E_z — the average over noise samples
max_D — the discriminator tries to make this large (high on real data, low on fakes)
min_G — the generator tries to make it small (fakes that D rates as real)
:::

In practice the generator maximizes log D(G(z)) instead (the "non-saturating" loss), which gives stronger gradients early in training. GANs produce sharp images in a single forward pass, but training is a delicate balancing act: it can oscillate or suffer **mode collapse**, where the generator produces only a few kinds of output.` },
      { type: 'text', title: 'Diffusion models', md: `Diffusion models learn to **undo noise**. The **forward process** gradually adds Gaussian noise to a clean example over T steps (often 1,000) until nothing but noise is left. It has a closed form, so you can jump straight to any step t:

$$ x_t = √(ᾱ_t) · x_0 + √(1 − ᾱ_t) · ε,          ε ~ N(0, I) $$
$$ ᾱ_t = (1 − β_1)(1 − β_2) ⋯ (1 − β_t) $$
:::where
x_0 — the clean data (for instance an image)
x_t — the noisy version after t steps
β_t — "beta": the small amount of noise added at step t (the noise schedule, e.g. from 0.0001 up to 0.02)
ᾱ_t — "alpha bar": the fraction of the original signal's variance that survives after t steps
ε — standard Gaussian noise with the same shape as x_0
I — the identity matrix: every value gets independent noise
:::

A neural network ε_θ(x_t, t) (usually a U-Net or a transformer) is trained to **predict the noise** that was added:

$$ L = ‖ ε − ε_θ(x_t, t) ‖² $$
:::where
ε_θ(x_t, t) — the network's guess of the noise, given the noisy input and the step number t
θ — the network's parameters
L — the mean squared error between the true and the predicted noise, for a random example and a random step t
:::

To **generate**, start from pure noise x_T and repeatedly subtract a little predicted noise, step by step, until a clean sample emerges. Text-to-image models condition the denoiser on a text embedding (via cross-attention), and **latent diffusion** (Stable Diffusion) runs the whole process in a VAE's compact latent space to save compute.` },
      { type: 'code', title: 'Forward diffusion on a 1-D signal', code: `
import numpy as np
rng = np.random.default_rng(0)
T = 1000
betas = np.linspace(1e-4, 0.02, T)                # the noise schedule
alpha_bar = np.cumprod(1 - betas)                  # how much of the signal survives after each step
x0 = np.sin(np.linspace(0, 4 * np.pi, 200))        # a clean 1-D "image"

for t in [0, 50, 200, 500, 999]:
    eps = rng.normal(size=x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    corr = np.corrcoef(x0, xt)[0, 1]
    print(f"t={t:4d}  signal weight {np.sqrt(alpha_bar[t]):.3f}  noise weight {np.sqrt(1 - alpha_bar[t]):.3f}  correlation with x0 {corr:.3f}")

t = 500                                            # a perfect noise predictor recovers x0 exactly:
eps = rng.normal(size=x0.shape)
xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
x0_hat = (xt - np.sqrt(1 - alpha_bar[t]) * eps) / np.sqrt(alpha_bar[t])
print("max error when the true noise is known:", float(np.abs(x0_hat - x0).max()))
`, after: `By the last step the signal is essentially gone. Learning to predict the added noise is learning to reverse this destruction, which is all a diffusion model needs to turn random noise into new data.` },
      { type: 'check', question: { type: 'tf', q: 'A diffusion model generates data by starting from pure noise and denoising it over many steps.', answer: true, explain: 'Each step removes a little of the noise the network predicts.' } },
      { type: 'text', title: "Evaluating generative models", md: `There is no single right answer to compare a generated image with, so generative models are judged on whole sets of samples. The most common metric, the **Fréchet Inception Distance (FID)**, embeds real and generated images with a pretrained image network, fits a Gaussian to each set of embeddings, and measures the distance between the two Gaussians:

$$ FID = ‖μ_r − μ_g‖² + Tr( Σ_r + Σ_g − 2 (Σ_r Σ_g)^(1/2) ) $$
:::where
μ_r, μ_g — the mean embeddings of the real and of the generated images
Σ_r, Σ_g — the covariance matrices of the two sets of embeddings
Tr — the trace: the sum of the diagonal entries
(Σ_r Σ_g)^(1/2) — the matrix square root of their product
:::

Lower is better. The first term checks that generated images are like real ones on average; the second that they are as varied, so a model that produces one perfect image over and over still scores badly. FID needs thousands of samples to be stable and depends on the embedding network, so compare models only under identical settings. Complementary metrics separate **precision** (are the samples realistic?) from **recall** (do they cover all kinds of real data?), and text-to-image models are also checked with image–text similarity scores and human preference ratings.` },
      { type: 'code', title: "The Fréchet distance between two sets of features", code: `import numpy as np
rng = np.random.default_rng(0)

def sqrtm_psd(M):
    vals, vecs = np.linalg.eigh((M + M.T) / 2)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T

def frechet(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root_a = sqrtm_psd(Sa)
    cross = sqrtm_psd(root_a @ Sb @ root_a)            # has the same trace as (Sa Sb)^(1/2)
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(Sa + Sb - 2 * cross))

real = rng.normal(0, 1, (5000, 16))                    # features of the "real" images
matching = rng.normal(0, 1, (5000, 16))                # a model that matches the distribution
shifted = rng.normal(0.5, 1, (5000, 16))               # samples that are systematically off
collapsed = rng.normal(0, 0.2, (5000, 16))             # samples with far too little variety
for name, fake in [("matching", matching), ("shifted", shifted), ("mode-collapsed", collapsed)]:
    print(f"{name:<15} Fréchet distance {frechet(real, fake):6.2f}")
print("lack of variety is penalised:", frechet(real, collapsed) > 5 * frechet(real, matching))`, after: `Samples drawn from the same distribution as the real ones score about 0.04 (not exactly 0, because both are finite samples). Shifting every feature by half a standard deviation costs about 4, all from the difference of the means. The collapsed samples have exactly the right mean but a fifth of the spread, and score worst of all, about 10: the covariance term catches the missing variety that a comparison of averages would miss.`, predict: {"type":"mcq","q":"Generated features have the right mean but only a fifth of the real spread. Will the Fréchet distance notice?","options":["No: only the means are compared","Yes: the covariance term penalises the missing variety","Only if the means differ","It becomes negative"],"answer":1,"explain":"The trace term compares the covariances, so samples that cluster too tightly are penalised even when their average is perfect.","expect":["lack of variety is penalised: True"]} }
    ],
    when: `- **Diffusion** for state-of-the-art image, audio and video generation and editing.
- **VAEs** for smooth latent spaces, compression, anomaly detection, and as the encoder/decoder inside latent diffusion.
- **GANs** when you need very fast single-step generation in a narrow domain (faces, super-resolution, style transfer).
- **Autoregressive models** for text and other sequences.
- Plain autoencoders for denoising, compression and unsupervised features.`,
    pitfalls: `- Judging generators only by eye: measure quality and diversity (e.g. FID) and check for memorized training examples.
- GAN training collapse or oscillation; changing several hyperparameters at once.
- A VAE KL term that is too strong, which makes the model ignore the latent code (blurry, average outputs).
- Using too few sampling steps for a diffusion model trained for many, without a matching sampler.
- Ignoring licensing, consent and misuse risks of training data and generated content.`,
    takeaways: [
      'Generative models learn p(x) so they can sample new, realistic data.',
      'Autoencoders compress through a bottleneck; VAEs add a KL term so the latent space can be sampled.',
      'Reparameterization: z = μ + σ·ε makes sampling differentiable.',
      'GANs pit a generator against a discriminator: sharp results, unstable training.',
      'Diffusion: noise data with a fixed schedule, train a network to predict the noise, and generate by iterative denoising.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the bottleneck of an autoencoder force the network to do?', options: ['Memorize every pixel', 'Keep only the most important structure of the input in a short code', 'Classify the input', 'Add noise'], answer: 1, explain: 'A small code can only hold the essential information.' },
      { type: 'mcq', q: 'What is mode collapse in GANs?', options: ['The discriminator stops training', 'The generator produces only a few kinds of output instead of the full variety of the data', 'The latent space becomes smooth', 'The loss becomes negative'], answer: 1, explain: 'It fools the discriminator with a narrow set of samples.' },
      { type: 'fill', q: 'In a VAE, μ = 2, σ = 0.5 and the sampled noise ε = −1. What is z = μ + σ·ε?', answer: ['1.5'], explain: '2 + 0.5 × (−1) = 1.5.' },
      { type: 'mcq', q: 'What is a diffusion model\'s network trained to predict?', options: ['The class of the image', 'The noise that was added to the clean example', 'The next pixel', 'The discriminator\'s output'], answer: 1, explain: 'Predicting the noise is equivalent to learning to denoise.' },
      { type: 'tf', q: 'Latent diffusion runs the diffusion process in a compressed latent space to save computation.', answer: true, explain: 'A VAE encodes images into small latents; diffusion happens there; the decoder produces the final image.' },
      {"type":"mcq","q":"A model generates one perfect image over and over. Why does FID still rate it badly?","options":["FID ignores realism","Its covariance term compares the variety of the generated and real images","FID only looks at colours","Repeated images crash the metric"],"answer":1,"explain":"Identical samples have zero spread, far from the spread of real images, so the trace term is large."},
      {"type":"tf","q":"FID scores computed with different embedding networks or sample sizes can be compared directly.","answer":false,"explain":"FID depends on the feature extractor and is biased by the number of samples, so only scores computed under identical settings are comparable."}
    ],
    problems: [
      {
        id: 'reparameterization', title: 'VAE Sampling and KL', difficulty: 'easy',
        prompt: `VAE encoders usually output the **log-variance** instead of σ (it can be any real number). Implement:

- \`sample_latent(mu, log_var, eps)\` → z = μ + exp(½ · log_var) · ε
- \`kl_divergence(mu, log_var)\` → the KL to the standard normal, summed over the **last** axis (one value per example for 2-D input):

$$ KL = ½ Σ_j ( μ_j² + exp(lv_j) − 1 − lv_j ) $$
:::where
μ_j — the mean for latent dimension j
lv_j — the log-variance log σ_j² for dimension j, so exp(lv_j) = σ_j²
Σ_j — the sum over the latent dimensions (the last axis)
:::

\`\`\`text
sample_latent(np.array([1.0]), np.array([0.0]), np.array([2.0]))  ->  [3.0]
kl_divergence(np.zeros(4), np.zeros(4))                            ->  0.0
\`\`\``,
        starter: `
import numpy as np

def sample_latent(mu, log_var, eps):
    # TODO
    pass

def kl_divergence(mu, log_var):
    # TODO
    pass
`,
        solution: `
import numpy as np

def sample_latent(mu, log_var, eps):
    return np.asarray(mu) + np.exp(0.5 * np.asarray(log_var)) * np.asarray(eps)

def kl_divergence(mu, log_var):
    mu, log_var = np.asarray(mu, dtype=float), np.asarray(log_var, dtype=float)
    return 0.5 * np.sum(mu ** 2 + np.exp(log_var) - 1 - log_var, axis=-1)
`,
        hints: ['σ = exp(½ · log_var).', 'Use `np.sum(..., axis=-1)` so the function works for one example or a batch.'],
        explain: `Predicting log-variance keeps σ positive without any constraint on the network's output. The KL is exactly zero only when μ = 0 and log_var = 0 (σ = 1) in every dimension.`,
        tests: [
          { call: 'sample_latent(np.array([1.0]), np.array([0.0]), np.array([2.0]))', expect: 'np.array([3.0])' },
          { call: 'kl_divergence(np.zeros(4), np.zeros(4))', expect: '0.0' },
          { name: 'known value', call: 'kl_divergence(np.array([1.0, 0.0]), np.array([0.0, np.log(0.25)]))', expect: '0.5 * (1.0 + (0.25 - 1 - np.log(0.25)))' },
          { name: 'batch of examples', setup: 'rng = np.random.default_rng(0)\nmu = rng.normal(size=(5, 3)); lv = rng.normal(size=(5, 3))', call: 'kl_divergence(mu, lv)', hidden: true },
          { name: 'samples have the right statistics', setup: 'rng = np.random.default_rng(1)\neps = rng.normal(size=200000)', code: 'z = sample_latent(2.0, np.log(9.0), eps)\nassert abs(z.mean() - 2.0) < 0.05 and abs(z.std() - 3.0) < 0.05', hidden: true }
        ]
      },
      {
        id: 'forward-diffusion', title: 'Forward Diffusion', difficulty: 'medium',
        prompt: `Implement the pieces of a diffusion model's noising process (t is a 0-based index into the schedule):

- \`linear_betas(T, start=1e-4, end=0.02)\` → \`np.linspace(start, end, T)\`
- \`alpha_bars(betas)\` → the cumulative product of (1 − β): element t is (1 − β_0)(1 − β_1)…(1 − β_t)
- \`add_noise(x0, t, alpha_bar, eps)\` → x_t = √(ᾱ_t) · x0 + √(1 − ᾱ_t) · eps
- \`predict_x0(xt, t, alpha_bar, eps_hat)\` → the clean signal implied by a noise prediction: (x_t − √(1 − ᾱ_t) · eps_hat) / √(ᾱ_t)

\`\`\`text
alpha_bars(np.array([0.1, 0.5]))  ->  [0.9, 0.45]
predict_x0(add_noise(x0, t, ab, eps), t, ab, eps)  ->  x0   (exactly, for the true noise)
\`\`\``,
        starter: `
import numpy as np

def linear_betas(T, start=1e-4, end=0.02):
    # TODO
    pass

def alpha_bars(betas):
    # TODO
    pass

def add_noise(x0, t, alpha_bar, eps):
    # TODO
    pass

def predict_x0(xt, t, alpha_bar, eps_hat):
    # TODO
    pass
`,
        solution: `
import numpy as np

def linear_betas(T, start=1e-4, end=0.02):
    return np.linspace(start, end, T)

def alpha_bars(betas):
    return np.cumprod(1 - np.asarray(betas, dtype=float))

def add_noise(x0, t, alpha_bar, eps):
    return np.sqrt(alpha_bar[t]) * np.asarray(x0) + np.sqrt(1 - alpha_bar[t]) * np.asarray(eps)

def predict_x0(xt, t, alpha_bar, eps_hat):
    return (np.asarray(xt) - np.sqrt(1 - alpha_bar[t]) * np.asarray(eps_hat)) / np.sqrt(alpha_bar[t])
`,
        hints: ['`np.cumprod` computes running products.', '`predict_x0` just solves the add_noise equation for x0.'],
        explain: `During training the network sees x_t and must output eps; \`predict_x0\` shows why that is enough: a good noise estimate gives you an estimate of the clean data, and samplers use it to take each denoising step.`,
        tests: [
          { call: 'alpha_bars(np.array([0.1, 0.5]))', expect: 'np.array([0.9, 0.45])' },
          { name: 'schedule', call: 'linear_betas(5)', expect: 'np.linspace(1e-4, 0.02, 5)' },
          { name: 'noising a signal', setup: 'ab = alpha_bars(linear_betas(100))\nx0 = np.array([1.0, -1.0, 0.5])\neps = np.array([0.3, 0.1, -0.2])', call: 'add_noise(x0, 40, ab, eps)' },
          { name: 'the true noise recovers x0', setup: 'rng = np.random.default_rng(0)\nab = alpha_bars(linear_betas(1000))\nx0 = rng.normal(size=(4, 4)); eps = rng.normal(size=(4, 4))', code: 'xt = add_noise(x0, 700, ab, eps)\nassert np.allclose(predict_x0(xt, 700, ab, eps), x0)', hidden: true },
          { name: 'signal fades over time', code: 'ab = alpha_bars(linear_betas(1000))\nassert np.all(np.diff(ab) < 0) and ab[0] > 0.999 and ab[-1] < 0.001', hidden: true }
        ]
      },
      {
        id: 'gan-losses', title: 'GAN Losses', difficulty: 'easy',
        prompt: `\`d_real\` and \`d_fake\` are arrays of discriminator outputs (probabilities of "real") for a batch of real and a batch of generated samples. First clip both to [1e-7, 1 − 1e-7] so no log ever sees 0. Then implement \`gan_losses(d_real, d_fake)\` → \`(d_loss, g_loss)\` as floats:

- discriminator loss: −mean(log d_real) − mean(log(1 − d_fake))
- generator loss (non-saturating): −mean(log d_fake)

\`\`\`text
gan_losses(np.array([0.5]), np.array([0.5]))  ->  (2·ln 2, ln 2) ≈ (1.386, 0.693)
\`\`\``,
        starter: `
import numpy as np

def gan_losses(d_real, d_fake):
    # TODO
    pass
`,
        solution: `
import numpy as np

def gan_losses(d_real, d_fake):
    d_real = np.clip(np.asarray(d_real, dtype=float), 1e-7, 1 - 1e-7)
    d_fake = np.clip(np.asarray(d_fake, dtype=float), 1e-7, 1 - 1e-7)
    d_loss = -np.mean(np.log(d_real)) - np.mean(np.log(1 - d_fake))
    g_loss = -np.mean(np.log(d_fake))
    return float(d_loss), float(g_loss)
`,
        hints: ['`np.clip(x, 1e-7, 1 - 1e-7)` keeps the logs finite.', 'Both losses are binary cross-entropies.'],
        explain: `When the discriminator is fooled (d_fake near 1) the generator loss is small; when it confidently rejects fakes, −log d_fake is large and gives the generator a strong gradient: the reason for the non-saturating form.`,
        tests: [
          { call: 'gan_losses(np.array([0.5]), np.array([0.5]))', expect: '(2 * np.log(2), np.log(2))' },
          { name: 'a confident, correct discriminator', call: 'gan_losses(np.array([0.99, 0.95]), np.array([0.01, 0.05]))' },
          { name: 'extreme outputs stay finite', call: 'gan_losses(np.array([1.0, 0.0]), np.array([0.0, 1.0]))', hidden: true },
          { name: 'random batch', setup: 'rng = np.random.default_rng(0)\ndr = rng.uniform(0.3, 1.0, 16); df = rng.uniform(0.0, 0.7, 16)', call: 'gan_losses(dr, df)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Kingma & Welling, Auto-Encoding Variational Bayes (VAE)', url: 'https://arxiv.org/abs/1312.6114' },
      { title: 'Goodfellow et al., Generative Adversarial Nets', url: 'https://arxiv.org/abs/1406.2661' },
      { title: 'Ho et al., Denoising Diffusion Probabilistic Models', url: 'https://arxiv.org/abs/2006.11239' },
      { title: 'Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models', url: 'https://arxiv.org/abs/2112.10752' }
    ]
  },

  // =====================================================================================
  {
    id: 'vision-transformers', branch: 'spec', title: 'Vision Transformers & Multimodal Models', short: 'Vision Transformers', level: 3, minutes: 40, prereqs: ['cnn', 'transformer-block'],
    tags: ['vision transformer', 'vit', 'patches', 'image classification', 'clip', 'contrastive learning', 'multimodal', 'vision-language models'],
    summary: 'Applying transformers to images: cutting an image into patches, patch embeddings with a class token and position embeddings, how ViTs compare with CNNs, and multimodal models (CLIP\'s contrastive image–text training and vision-language models that read images).',
    why: `The transformer turned out to be a general-purpose architecture: treat an image as a sequence of patches and the same blocks that power GPT can classify, detect and segment, and, joined to a language model, let assistants read screenshots, charts and photos. CLIP-style image–text embeddings drive image search and serve as the text encoders inside image generators.`,
    objectives: [
      'Turn an image into a sequence of patch tokens and compute the sequence length',
      'Describe the ViT architecture: patch embedding, class token, position embeddings, encoder blocks, classification head',
      'Compare ViTs and CNNs: inductive biases, data needs and compute',
      'Explain CLIP\'s contrastive objective and how vision-language models feed images into an LLM'
    ],
    sections: [
      { type: 'text', title: 'An image is worth 16 × 16 words', md: `A **Vision Transformer (ViT)** turns an image into a sequence of tokens:

1. Cut the H × W image (with C colour channels) into non-overlapping P × P **patches**.
2. Flatten each patch into a vector and project it with one learned matrix to the model width d: the **patch embedding** (equivalent to a convolution with kernel size and stride P).
3. Prepend a learned **[CLS] token** and add learned **position embeddings**.
4. Run a standard transformer **encoder** (bidirectional attention: every patch can attend to every other).
5. Classify from the final [CLS] vector with a small MLP head.

$$ N = (H / P) · (W / P)          patch vector length = P · P · C $$
:::where
H, W — the image height and width in pixels
C — the number of channels (3 for RGB)
P — the patch size, often 16
N — the number of patches, which is the sequence length the transformer sees (plus one for [CLS])
:::

A 224 × 224 RGB image with P = 16 gives 14 × 14 = 196 patches of 768 numbers each, plus the [CLS] token: a 197-token sequence.` },
      { type: 'code', title: 'Patchify and embed an image in NumPy', code: `
import numpy as np
rng = np.random.default_rng(0)
H = W = 32; C = 3; P = 8; d = 64
img = rng.random((H, W, C))

# (H, W, C) -> (H/P, P, W/P, P, C) -> (H/P, W/P, P, P, C) -> (N, P*P*C)
patches = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * C)
print("image", img.shape, "->", patches.shape[0], "patches of", patches.shape[1], "numbers")
print("patch 1 is the top row, second block:", np.allclose(patches[1], img[0:8, 8:16].reshape(-1)))

W_embed = rng.normal(0, 0.02, (P * P * C, d))         # the patch-embedding projection
cls = rng.normal(0, 0.02, (1, d))                      # learned [CLS] token
pos = rng.normal(0, 0.02, (patches.shape[0] + 1, d))   # learned position embeddings
tokens = np.vstack([cls, patches @ W_embed]) + pos
print("token sequence for the transformer:", tokens.shape)
` },
      { type: 'check', question: { type: 'fill', q: 'A 384 × 384 RGB image is split into 16 × 16 patches. How many patches are there?', answer: ['576'], explain: '(384 / 16)² = 24² = 576.' } },
      { type: 'text', title: 'ViTs versus CNNs', md: `| | CNN | ViT |
| --- | --- | --- |
| Built-in assumptions | locality and translation equivariance (strong inductive bias) | almost none: any patch can attend to any other from the first layer |
| Data needed | works well on small and medium datasets | needs large datasets, heavy augmentation or distillation |
| Scaling | good | excellent with more data and compute |
| Cost | linear in the number of pixels | attention is quadratic in the number of patches |

Strong inductive biases help when data is scarce; with enough data, ViTs learn those regularities themselves and keep improving. Modern designs mix both ideas: Swin uses attention within local windows, ConvNeXt modernizes CNNs with transformer-style tricks. The patch size trades detail for compute: smaller patches mean more tokens.` },
      { type: 'text', title: "Making ViTs practical", md: `The original ViT needed hundreds of millions of labelled images to beat CNNs, because it lacks their built-in assumptions. Later work closed the gap:
- **DeiT** trained ViTs on ImageNet alone with strong augmentation (random crops, mixed images, colour changes), regularisation, and distillation from a CNN teacher.
- **Self-supervised pretraining** (MAE, DINO; see [Self-Supervised Representation Learning](#/concept/self-supervised)) lets ViTs learn from unlabelled images.
- **Swin Transformer** restores some locality: attention runs inside small windows of patches that shift from one layer to the next, and patches are merged as the network gets deeper, producing a CNN-like pyramid of resolutions that suits detection and segmentation.
- **Modern CNNs** (ConvNeXt) showed that convolutional networks updated with transformer-era training recipes remain competitive.

The patch size is the main cost knob: halving it quadruples the number of tokens and multiplies the cost of global attention by sixteen.` },
      { type: 'code', title: "Global attention versus windowed attention", code: `def attention_pairs(image_side, patch, window=None):
    grid = image_side // patch                     # patches per side
    tokens = grid * grid
    if window is None:
        return tokens, tokens * tokens             # every patch attends to every patch
    return tokens, tokens * window * window        # each patch attends within its window x window block

for patch in (16, 8, 4):
    tokens, pairs = attention_pairs(224, patch)
    print(f"patch {patch:2d}: {tokens:5d} tokens, global attention {pairs:>12,} pairs")
tokens, pairs = attention_pairs(224, 4, window=7)
print(f"patch  4 with 7x7 windows (Swin-style): {pairs:,} pairs")
print("windows cut the cost 64-fold:", attention_pairs(224, 4)[1] // pairs == 64)`, after: `Going from 16-pixel to 4-pixel patches multiplies the work of global attention by 256, from about 38 thousand pairs to almost 10 million. Windows of 7 × 7 patches bring it down 64-fold, so the cost grows only linearly with the image area, and the shifted windows of the next layer let information cross the window borders.`, predict: {"type":"mcq","q":"Patch size 4 on a 224 × 224 image gives 3,136 tokens. Attending only within 7 × 7 windows instead of globally cuts the number of attention pairs by what factor?","options":["7","49","64","3,136"],"answer":2,"explain":"Global: 3,136 × 3,136 pairs; windowed: 3,136 × 49. The ratio is 3,136 / 49 = 64.","expect":["windows cut the cost 64-fold: True"]} },
      { type: 'text', title: 'CLIP: learning from images and captions', md: `**CLIP** trains an image encoder (a ViT) and a text encoder (a transformer) together on hundreds of millions of image–caption pairs from the web, so that matching images and texts land close together in one shared embedding space. Within a batch of B pairs, every image is compared with every caption:

$$ logits_ij = ( u_i · v_j ) / τ $$
$$ loss = ½ [ CE_rows(logits) + CE_columns(logits) ],   correct answers on the diagonal $$
:::where
u_i — the normalized embedding of image i
v_j — the normalized embedding of caption j
u_i · v_j — their cosine similarity
τ — "tau": a temperature (about 0.07) that sharpens the softmax
CE_rows — cross-entropy where each image must pick its own caption among the B captions; CE_columns does the same for each caption
:::

This **contrastive** objective needs no hand-made labels. It enables **zero-shot classification**: embed prompts such as "a photo of a dog" and "a photo of a cat", and pick the one closest to the image. CLIP-style models power image search and dataset filtering, and their text encoders guide image generators.` },
      { type: 'code', title: 'A CLIP-style contrastive loss', code: `
import numpy as np
rng = np.random.default_rng(0)

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def cross_entropy_diag(logits):                    # the correct "class" for row i is column i
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -np.mean(np.diag(logp))

def clip_loss(img, txt, tau=0.07):
    logits = normalize(img) @ normalize(txt).T / tau     # (B, B) table of similarities
    return (cross_entropy_diag(logits) + cross_entropy_diag(logits.T)) / 2

B, d = 8, 32
img = rng.normal(size=(B, d))
print("a uniform guess would score ln B =", round(float(np.log(B)), 3))
print("unrelated embeddings:    loss", round(float(clip_loss(img, rng.normal(size=(B, d)))), 3))
print("matching pairs aligned:  loss", round(float(clip_loss(img, img + 0.3 * rng.normal(size=(B, d)))), 3))
`, after: `Unrelated embeddings score even worse than a uniform guess: the low temperature turns small accidental similarities into confident wrong matches. When each image sits next to its own caption, the loss drops to almost zero.` },
      { type: 'text', title: 'Vision-language models', md: `Multimodal assistants (LLaVA, GPT-4o, Claude, Gemini) let an LLM read images:

\`\`\`text
image -> vision encoder (a ViT, often CLIP-trained) -> patch embeddings
      -> small projector (MLP) into the LLM's token-embedding space -> "image tokens"
prompt = [image tokens] + [text tokens]  ->  the LLM attends to both like ordinary words
\`\`\`

Because an image becomes a few hundred to a few thousand tokens, images cost tokens and context window just like text; higher resolutions mean more patches. The same recipe extends to audio and video frames.` },
      { type: 'check', question: { type: 'tf', q: 'A Vision Transformer needs no convolutions at all: images are split into patches, and attention lets every patch interact with every other.', answer: true, explain: 'The patch embedding is just a linear projection (equivalent to a strided convolution).' } }
    ],
    when: `- ViTs for image classification, detection and segmentation when you have plenty of data or a pretrained backbone; CNNs for small datasets, edge devices and latency-critical work.
- CLIP-style embeddings for image search, zero-shot classification and dataset curation.
- Vision-language models for documents, screenshots, charts and visual question answering.`,
    pitfalls: `- Training a ViT from scratch on a small dataset (use a pretrained model).
- Forgetting that the patch grid fixes the input resolution: position embeddings must be interpolated for other sizes.
- Underestimating attention cost at high resolution.
- Expecting zero-shot CLIP to handle counting, fine-grained text or precise spatial relations reliably.
- Treating text read from images as trusted instructions (prompt injection via images).`,
    takeaways: [
      'ViT: image → P × P patches → linear embedding + [CLS] + positions → transformer encoder.',
      'Number of tokens N = (H/P)·(W/P); attention cost grows with N².',
      'CNNs carry strong inductive biases (good with little data); ViTs scale better with lots of data.',
      'CLIP aligns images and text with a contrastive loss, enabling zero-shot classification and search.',
      'Vision-language models project image patches into an LLM\'s embedding space as "image tokens".'
    ],
    quiz: [
      { type: 'mcq', q: 'What is the role of the [CLS] token in a ViT?', options: ['It marks the end of the image', 'A learned token whose final vector summarizes the image for classification', 'It stores the position embeddings', 'It is the first patch of the image'], answer: 1, explain: 'It attends to all patches; the classifier reads it.' },
      { type: 'mcq', q: 'Why do ViTs typically need more training data than CNNs?', options: ['They have fewer parameters', 'They lack built-in locality and translation biases and must learn them from data', 'They cannot use GPUs', 'They only work on text'], answer: 1, explain: 'Weaker inductive bias means more data to learn the same regularities.' },
      { type: 'mcq', q: 'In CLIP\'s contrastive loss, which pairs should get high similarity?', options: ['All pairs in the batch', 'Each image with its own caption (the diagonal)', 'Images with images', 'Random pairs'], answer: 1, explain: 'Matching pairs are the "correct class"; all others are negatives.' },
      { type: 'tf', q: 'Halving the patch size of a ViT quadruples the number of patches.', answer: true, explain: 'N = (H/P)(W/P): halving P doubles both factors.' },
      { type: 'mcq', q: 'How does a vision-language model give an image to the LLM?', options: ['It converts the image to a caption first, always', 'It encodes patches with a vision encoder and projects them into the LLM\'s token-embedding space', 'It feeds raw pixels as numbers in the text', 'It cannot'], answer: 1, explain: 'The projected patch embeddings act as image tokens.' },
      {"type":"mcq","q":"Halving the patch size of a ViT multiplies the cost of global attention by about…","options":["2","4","16","256"],"answer":2,"explain":"Half the patch size gives four times the tokens, and global attention's cost grows with the square of the number of tokens: 4² = 16."},
      {"type":"mcq","q":"What did DeiT show?","options":["ViTs need billions of images","With strong augmentation, regularisation and distillation, ViTs can be trained well on ImageNet alone","CNNs cannot be distilled","Patches must be 32 pixels"],"answer":1,"explain":"DeiT closed the data gap with training recipes rather than more data, making ViTs practical for ordinary datasets."}
    ],
    problems: [
      {
        id: 'image-to-patches', title: 'Patchify an Image', difficulty: 'medium',
        prompt: `Implement:

- \`patchify(img, P)\`: \`img\` has shape (H, W, C). Raise \`ValueError\` if H or W is not divisible by P. Return an array of shape (N, P·P·C), with patches in reading order (left to right, then top to bottom) and each patch flattened in (row, column, channel) order, i.e. patch k equals \`img[r*P:(r+1)*P, c*P:(c+1)*P].reshape(-1)\` for k = r·(W/P) + c.
- \`unpatchify(patches, H, W, C, P)\`: the exact inverse.

Hint: reshape to (H/P, P, W/P, P, C), transpose to (H/P, W/P, P, P, C), then reshape.`,
        starter: `
import numpy as np

def patchify(img, P):
    # TODO
    pass

def unpatchify(patches, H, W, C, P):
    # TODO
    pass
`,
        solution: `
import numpy as np

def patchify(img, P):
    img = np.asarray(img)
    H, W, C = img.shape
    if H % P or W % P:
        raise ValueError("image size must be divisible by the patch size")
    return img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * C)

def unpatchify(patches, H, W, C, P):
    return np.asarray(patches).reshape(H // P, W // P, P, P, C).transpose(0, 2, 1, 3, 4).reshape(H, W, C)
`,
        hints: ['Reshape first, so each row splits into (block row, row inside the block).', 'The transpose brings the two block indices to the front.', 'For the inverse, undo the steps in reverse order.'],
        explain: `Reshape-transpose-reshape rearranges memory without loops: the same trick used for splitting attention heads. Real ViTs implement this step as a convolution with kernel size P and stride P, followed by flattening.`,
        tests: [
          { name: 'a 4×4 single-channel image', call: 'patchify(np.arange(16).reshape(4, 4, 1), 2)', expect: 'np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])' },
          { name: 'shape and one patch', setup: 'rng = np.random.default_rng(0)\nimg = rng.random((32, 48, 3))', code: 'p = patchify(img, 16)\nassert p.shape == (6, 768), p.shape\nassert np.allclose(p[4], img[16:32, 16:32].reshape(-1))' },
          { name: 'round trip', setup: 'rng = np.random.default_rng(1)\nimg = rng.random((24, 24, 3))', code: 'assert np.allclose(unpatchify(patchify(img, 8), 24, 24, 3, 8), img)', hidden: true },
          { name: 'not divisible', code: 'try:\n    patchify(np.zeros((10, 10, 3)), 4)\n    raise AssertionError("expected ValueError")\nexcept ValueError:\n    pass', hidden: true },
          { name: 'random image vs reference', setup: 'rng = np.random.default_rng(2)\nimg = rng.random((8, 12, 2))', call: 'patchify(img, 4)', hidden: true }
        ]
      },
      {
        id: 'clip-loss', title: 'CLIP Contrastive Loss', difficulty: 'medium',
        prompt: `\`img_emb\` and \`txt_emb\` are (B, d) arrays; row i of each belongs to the same image–caption pair. Implement \`clip_loss(img_emb, txt_emb, tau=0.07)\`:

1. Normalize every row to unit length.
2. logits = img @ txtᵀ / tau, a (B, B) matrix.
3. Image-to-text loss: the mean over rows of the cross-entropy of each row, where the correct column for row i is i (use a stable log-softmax over each row).
4. Text-to-image loss: the same on the transposed logits.
5. Return the average of the two as a float.

\`\`\`text
Identical, mutually orthogonal image/text pairs give a loss close to 0.
If every embedding is the same, all captions look equally likely and the loss is exactly ln B.
\`\`\``,
        starter: `
import numpy as np

def clip_loss(img_emb, txt_emb, tau=0.07):
    # TODO
    pass
`,
        solution: `
import numpy as np

def log_softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def clip_loss(img_emb, txt_emb, tau=0.07):
    I = np.asarray(img_emb, dtype=float)
    T = np.asarray(txt_emb, dtype=float)
    I = I / np.linalg.norm(I, axis=1, keepdims=True)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    logits = I @ T.T / tau
    loss_i = -np.mean(np.diag(log_softmax_rows(logits)))
    loss_t = -np.mean(np.diag(log_softmax_rows(logits.T)))
    return float((loss_i + loss_t) / 2)
`,
        hints: ['`np.diag` of the log-softmax matrix picks each row\'s correct column.', 'The text-to-image direction is the same computation on `logits.T`.'],
        explain: `Every other caption in the batch serves as a negative example, so bigger batches give a harder, more informative task: CLIP used batches of 32,768 pairs.`,
        tests: [
          { name: 'perfectly aligned orthogonal pairs', call: 'clip_loss(np.eye(4), np.eye(4))', expect: 'float(-np.log(np.exp(1 / 0.07) / (np.exp(1 / 0.07) + 3)))' },
          { name: 'identical embeddings give chance level', call: 'clip_loss(np.ones((5, 3)), np.ones((5, 3)))', expect: 'np.log(5)' },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nI = rng.normal(size=(6, 16)); T = rng.normal(size=(6, 16))', call: 'clip_loss(I, T)', hidden: true },
          { name: 'aligned pairs beat random pairs', setup: 'rng = np.random.default_rng(1)\nI = rng.normal(size=(8, 32))', code: 'aligned = clip_loss(I, I + 0.1 * rng.normal(size=I.shape))\nrandom_ = clip_loss(I, rng.normal(size=I.shape))\nassert aligned < random_', hidden: true },
          { name: 'scale invariance and temperature', setup: 'rng = np.random.default_rng(2)\nI = rng.normal(size=(4, 8)); T = rng.normal(size=(4, 8))', code: 'assert np.isclose(clip_loss(I, T, 0.5), clip_loss(3 * I, 0.2 * T, 0.5))', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Dosovitskiy et al., An Image is Worth 16x16 Words (ViT)', url: 'https://arxiv.org/abs/2010.11929' },
      { title: 'Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP)', url: 'https://arxiv.org/abs/2103.00020' },
      { title: 'Liu et al., Visual Instruction Tuning (LLaVA)', url: 'https://arxiv.org/abs/2304.08485' }
    ]
  },

  // =====================================================================================
  {
    id: 'recommender-systems', branch: 'spec', title: 'Recommender Systems', short: 'Recommenders', level: 2, minutes: 45, prereqs: ['embeddings', 'regularization'],
    tags: ['recommender systems', 'collaborative filtering', 'content-based', 'matrix factorization', 'embeddings', 'implicit feedback', 'cold start', 'two-tower', 'ranking', 'precision@k'],
    summary: 'Suggesting items people will like: content-based versus collaborative filtering, matrix factorization with user and item embeddings trained by SGD, implicit feedback and cold start, the retrieval → ranking pipeline used at scale, and how to evaluate recommendations.',
    why: `Recommenders decide much of what people see online: videos, products, songs, news, jobs, friends. They are among the most commercially important ML systems, and a clear application of embeddings: learn a vector for every user and every item so that their dot product predicts how much the user will like the item.`,
    objectives: [
      'Distinguish content-based and collaborative filtering',
      'Factorize a sparse rating matrix into user and item embeddings and train it with SGD and L2 regularization',
      'Handle implicit feedback and the cold-start problem',
      'Describe the retrieval → ranking pipeline and evaluate with precision@k and recall@k'
    ],
    sections: [
      { type: 'text', title: 'The problem', md: `Picture a huge **users × items** matrix of interactions: ratings, clicks, purchases, watch time. Almost all of it is empty: a typical user has touched a tiny fraction of the catalogue (often well under 1%). The task is to predict the missing entries, or simply to rank the items each user is most likely to want.

| Approach | Idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **Content-based** | recommend items similar to what the user liked, using item features (genre, text, image) | works for brand-new items; explainable | stays close to the user's past; needs good features |
| **Collaborative filtering** | users who behaved like you liked X | discovers non-obvious tastes; no features needed | can't handle new users/items (cold start) |
| **Hybrid** | combine both | best of both | more complex |` },
      { type: 'chart', title: "The long tail of item popularity", md: `200,000 simulated interactions spread over 2,000 items with a typical power-law popularity:`, after: `The top 1% of items (20 of them) collect 54% of all interactions and the top 20% collect 86%, while the items in the long tail have only a handful each. Collaborative filtering learns little about tail items, and a recommender that ranks by learned relevance tends to push the head even harder.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,12,13,15,17,19,22,25,28,32,37,42,48,54,62,70,80,91,103,118,134,152,173,197,224,255,290,329,375,426,485,551,627,714,812,923,1050,1195,1359,1546,1758,2000],"xLog":true,"yLog":true,"points":false,"series":[{"name":"interactions per item","y":[34056,15519,10132,7357,5916,4629,4009,3414,3030,2921,2227,2087,1693,1433,1304,1162,979,840,788,638,525,481,429,361,310,280,235,205,181,161,140,118,100,88,76,65,58,49,44,38,33,29,25,21,18,16,14,12,10,8,1]}],"xLabel":"item rank by popularity (log scale)","yLabel":"interactions (log scale)"} },
      { type: 'text', title: 'Matrix factorization', md: `Collaborative filtering's workhorse learns a k-dimensional embedding for every user and every item (k ≈ 10–200), plus biases:

$$ r̂_ui = μ + b_u + b_i + p_u · q_i $$
:::where
r̂_ui — the predicted rating of user u for item i
μ — the global average rating
b_u — user u's bias (some people rate everything high)
b_i — item i's bias (some items are liked by everyone)
p_u, q_i — the k-dimensional embeddings of user u and item i; their dot product measures how well the user's tastes match the item
:::

Train on the **observed** ratings only, with L2 regularization to avoid overfitting the sparse data:

$$ L = Σ_(u,i) observed ( r_ui − r̂_ui )² + λ ( ‖p_u‖² + ‖q_i‖² + b_u² + b_i² ) $$
:::where
r_ui — an observed (true) rating
Σ_(u,i) observed — a sum over the known ratings only; missing entries are ignored, not treated as zeros
λ — "lambda": the regularization strength
‖p_u‖² — the squared length of a user's embedding (likewise for items)
:::

Stochastic gradient descent visits one rating at a time. With the error e = r_ui − r̂_ui and learning rate η:

$$ p_u ← p_u + η ( e · q_i − λ p_u )          q_i ← q_i + η ( e · p_u − λ q_i ) $$
$$ b_u ← b_u + η ( e − λ b_u )          b_i ← b_i + η ( e − λ b_i ) $$
:::where
e — the prediction error for this rating (positive when the prediction was too low)
η — "eta": the learning rate
e · q_i — moves the user's embedding towards the item's when the rating was higher than predicted
Both embedding updates use the values from before this step.
:::

Nobody tells the model what the dimensions mean, yet they often line up with recognizable tastes (action versus romance, mainstream versus niche). This approach won much of the Netflix Prize.` },
      { type: 'code', title: 'Matrix factorization from scratch', timeoutSec: 90, code: `
import numpy as np
rng = np.random.default_rng(0)
n_users, n_items, k = 120, 80, 3
true_P, true_Q = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
R_full = 3 + 0.6 * true_P @ true_Q.T + rng.normal(0, 0.3, (n_users, n_items))   # hidden "true" ratings
mask = rng.random((n_users, n_items)) < 0.35                                       # only 35% are observed
obs = [(int(u), int(i), float(R_full[u, i])) for u, i in zip(*np.nonzero(mask))]
rng.shuffle(obs)
train, test = obs[: int(0.8 * len(obs))], obs[int(0.8 * len(obs)):]

mu = np.mean([r for _, _, r in train])
bu, bi = np.zeros(n_users), np.zeros(n_items)
P, Q = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
lr, reg = 0.02, 0.08

def rmse(data):
    return np.sqrt(np.mean([(r - (mu + bu[u] + bi[i] + P[u] @ Q[i])) ** 2 for u, i, r in data]))

print(f"always predicting the mean: test RMSE {np.sqrt(np.mean([(r - mu) ** 2 for _, _, r in test])):.3f}")
for epoch in range(25):
    for u, i, r in train:
        e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])
        P[u], Q[i] = P[u] + lr * (e * Q[i] - reg * P[u]), Q[i] + lr * (e * P[u] - reg * Q[i])
    if epoch % 5 == 4:
        print(f"epoch {epoch + 1:2d}: train RMSE {rmse(train):.3f}   test RMSE {rmse(test):.3f}")
`, after: `For the first few epochs the tiny random embeddings barely help; once they grow, the model captures the users' tastes and the test error falls well below the "predict the average" baseline.` },
      { type: 'check', question: { type: 'mcq', q: 'In matrix factorization, which entries of the rating matrix appear in the training loss?', options: ['All entries, with missing ones set to 0', 'Only the observed ratings', 'Only the missing entries', 'Only the diagonal'], answer: 1, explain: 'Missing does not mean "rated zero"; it means unknown.' } },
      { type: 'text', title: 'Implicit feedback and cold start', md: `Most real data is **implicit**: clicks, views, purchases, listening time. There are no explicit dislikes, not clicking an item doesn't mean you dislike it; you may never have seen it. Common approaches: treat interactions as positives and **sample** some unobserved items as likely negatives; weight observations by confidence (weighted matrix factorization); or train with a ranking loss such as **BPR**, which only asks that an item the user interacted with scores higher than a random one they didn't.

**Cold start**: a new user or item has no interactions, so collaborative filtering knows nothing about it. Fixes: content features (embed the item's text and images), popularity-based defaults, a few onboarding questions, and **bandit** exploration to learn quickly what newcomers like (see *Reinforcement Learning*).` },
      { type: 'text', title: 'Recommendation at scale', md: `Large platforms rank millions of items in milliseconds with a multi-stage pipeline:

\`\`\`text
millions of items --(1) candidate retrieval--> ~500 candidates --(2) ranking--> top 50 --(3) re-ranking--> page
\`\`\`

1. **Retrieval** with a **two-tower model**: one network embeds the user (history, context), another embeds each item, trained so that interacted pairs score high. Item embeddings are precomputed and searched with approximate nearest-neighbour search: the same machinery as RAG.
2. **Ranking**: a heavier model with rich features (user history, item statistics, time, device) predicts the probability of a click, purchase or long watch for each candidate.
3. **Re-ranking**: business rules, diversity, freshness, and removing items already seen.

**Feedback loops** are a real risk: recommendations shape what people see, which shapes the next training data. Popular items get more popular, and users can get trapped in filter bubbles, so production systems deliberately add exploration and diversity.` },
      { type: 'text', title: 'Evaluating recommenders', md: `Offline, hold out each user's **most recent** interactions (a time-based split, predicting the past from the future is cheating) and measure ranking quality:

$$ precision@k = |relevant ∩ top_k| / k          recall@k = |relevant ∩ top_k| / |relevant| $$
:::where
top_k — the first k items you recommended to the user
relevant — the held-out items the user actually interacted with
∩ — the intersection: items that are in both sets
|…| — the number of items in a set
precision@k — the share of the recommendations that were right
recall@k — the share of the user's relevant items that you found
:::

Ranking-aware metrics such as NDCG also reward putting hits near the top, and RMSE suits explicit ratings. Online, run **A/B tests** on what actually matters (clicks, watch time, purchases, retention) because offline gains don't always survive contact with real users.` },
      { type: 'text', title: "Beyond accuracy: popularity bias and feedback loops", md: `Interactions follow a **long tail**: a few blockbuster items collect a large share of all activity, and most of the catalogue is rarely touched (see the chart at the top). Two consequences:

- **Popularity is a strong baseline.** Recommending the most popular items a user hasn't seen yet is often surprisingly hard to beat on accuracy metrics. Always report it next to your model.
- **Recommenders amplify popularity.** Recommended items get clicked, which makes them look even more relevant in the next training run, while tail items never get the exposure they would need to prove themselves. Left alone, this **feedback loop** narrows what users see and starves new items.

That is why recommender teams track more than accuracy:

| Metric | Question it answers |
| --- | --- |
| catalogue coverage | what share of the items is ever recommended? |
| diversity | how different are the items within one list? |
| novelty | how far from the blockbusters are the recommendations? |
| calibration | does a list reflect the mix of the user's interests, say 70% drama and 30% comedy? |

Remedies include re-ranking for diversity, reserving slots for exploration (contextual bandits try uncertain items to learn about them), and correcting the training data for exposure with the same inverse-propensity idea used for position bias in [Learning to Rank](#/concept/learning-to-rank).` },
      { type: 'code', title: "A popularity baseline is hard to beat", code: `import numpy as np
rng = np.random.default_rng(0)
n_users, n_items = 1000, 500
popularity = 1 / np.arange(1, n_items + 1) ** 1.1             # a long tail: item 0 is the blockbuster
popularity /= popularity.sum()
history = [rng.choice(n_items, 21, replace=False, p=popularity) for _ in range(n_users)]
train = [set(h[:20].tolist()) for h in history]                # each user's first 20 items
test = [int(h[20]) for h in history]                            # the item they chose next

counts = np.zeros(n_items)
for items in train:
    counts[list(items)] += 1
ranked = np.argsort(-counts)                                    # items from most to least popular

def most_popular_unseen(seen, k=10):
    return [int(i) for i in ranked if i not in seen][:k]

recs = [most_popular_unseen(train[u]) for u in range(n_users)]
hit_pop = np.mean([test[u] in recs[u] for u in range(n_users)])
hit_rand = np.mean([test[u] in rng.choice([i for i in range(n_items) if i not in train[u]], 10, replace=False)
                    for u in range(n_users)])
coverage = len({i for r in recs for i in r})
print(f"hit rate@10, most popular unseen items: {hit_pop:.3f}")
print(f"hit rate@10, random unseen items:       {hit_rand:.3f}")
print(f"items ever recommended by the popularity baseline: {coverage} of {n_items}")
print("popularity beats random:", bool(hit_pop > 5 * hit_rand))`, after: `Without knowing anything about the user, the popularity baseline finds their next item about 30% of the time, roughly 15 times better than random. But it only ever recommends 27 of the 500 items: about 95% of the catalogue is invisible. A personalised model has to beat the first number without collapsing the second.`, predict: {"type":"mcq","q":"Users' choices follow a long tail. How will 'recommend the 10 most popular items the user hasn't seen' compare with 10 random unseen items?","options":["About the same","Far better","Worse","It cannot be evaluated"],"answer":1,"explain":"With a long tail, a large share of every user's next choices are popular items, so popularity alone is a strong predictor, although it shows the same few items to everyone.","expect":["popularity beats random: True"]} },
      { type: 'check', question: { type: 'tf', q: 'A two-tower retrieval model embeds users and items separately so item embeddings can be precomputed and searched quickly.', answer: true, explain: 'Only the user tower runs at request time.' } }
    ],
    when: `- Any product with a large catalogue and repeated user interactions: e-commerce, media, news, jobs, dating, ads.
- Content-based methods for new items and niche catalogues; collaborative filtering once you have interaction data; bandits for fresh content and new users.
- Two-tower retrieval + ranking when the catalogue is large.`,
    pitfalls: `- Treating missing entries as negative ratings.
- Random train/test splits that leak future interactions into training.
- Optimizing clicks alone and ending up with clickbait; ignoring diversity.
- Popularity bias and feedback loops that starve new and niche items.
- Ignoring cold start for new users and items.`,
    takeaways: [
      'Collaborative filtering learns from everyone\'s behaviour; content-based uses item features.',
      'Matrix factorization: r̂ = μ + b_u + b_i + p_u·q_i, trained on observed entries with L2 regularization.',
      'Implicit feedback needs negative sampling or ranking losses; cold start needs content features or exploration.',
      'At scale: two-tower retrieval → ranking model → re-ranking.',
      'Evaluate with time-based splits and precision/recall@k offline, A/B tests online.'
    ],
    quiz: [
      { type: 'mcq', q: 'What does the dot product p_u · q_i represent in matrix factorization?', options: ['The number of ratings', 'How well user u\'s latent tastes match item i\'s latent features', 'The global average', 'The item\'s popularity'], answer: 1, explain: 'Aligned embeddings give a high score.' },
      { type: 'mcq', q: 'What is the cold-start problem?', options: ['Training on a cold GPU', 'New users or items have no interaction history, so collaborative filtering cannot score them', 'Slow model start-up', 'Ratings that are too low'], answer: 1, explain: 'No data means no learned embedding.' },
      { type: 'fill', q: 'You recommended 5 items; 2 of them are among the user\'s 4 relevant items. What is precision@5?', answer: ['0.4', '0.40', '2/5'], explain: '2 / 5 = 0.4 (recall@5 would be 2/4 = 0.5).' },
      { type: 'tf', q: 'For implicit feedback, an item the user never clicked can safely be treated as disliked.', answer: false, explain: 'They may simply never have seen it.' },
      { type: 'mcq', q: 'Why use a time-based split to evaluate a recommender?', options: ['It is faster', 'A random split leaks future behaviour into training, making offline results look better than they will be', 'Time-based splits have more data', 'It is required by law'], answer: 1, explain: 'In production you always predict the future from the past.' },
      {"type":"mcq","q":"What is catalogue coverage?","options":["The share of users who receive recommendations","The share of items that are ever recommended to anyone","The accuracy of the top recommendation","The number of ratings per user"],"answer":1,"explain":"Coverage measures how much of the catalogue the recommender actually exposes; a popularity baseline scores very low on it."},
      {"type":"tf","q":"Training a recommender only on clicks from its own past recommendations can reinforce popularity bias.","answer":true,"explain":"Items it recommended collect clicks and look better in the next round, while items it never showed get no chance: a feedback loop."}
    ],
    problems: [
      {
        id: 'mf-predict-sgd', title: 'Matrix Factorization with SGD', difficulty: 'medium',
        prompt: `Implement:

- \`predict(mu, bu, bi, P, Q, u, i)\` → μ + bu[u] + bi[i] + P[u] · Q[i] (a float).
- \`sgd_epoch(ratings, mu, bu, bi, P, Q, lr, reg)\`: work on **copies** of bu, bi, P and Q and return them as \`(bu, bi, P, Q)\`. Visit \`ratings\` (a list of \`(u, i, r)\`) in order. For each rating compute e = r − predict(...), then:
  - bu[u] += lr · (e − reg · bu[u]) and bi[i] += lr · (e − reg · bi[i])
  - P[u] and Q[i] are updated **together**, both from their values before this rating: P[u] += lr · (e · Q[i] − reg · P[u]) and Q[i] += lr · (e · P[u] − reg · Q[i])

\`\`\`text
One rating (0, 0, 5.0) with mu = 3, zero biases, P = [[1, 0]], Q = [[1, 1]], lr = 0.1, reg = 0:
e = 5 − (3 + 1) = 1   ->  bu = [0.1], bi = [0.1], P = [[1.1, 0.1]], Q = [[1.1, 1.0]]
\`\`\``,
        starter: `
import numpy as np

def predict(mu, bu, bi, P, Q, u, i):
    # TODO
    pass

def sgd_epoch(ratings, mu, bu, bi, P, Q, lr, reg):
    bu, bi, P, Q = (np.array(a, dtype=float) for a in (bu, bi, P, Q))
    # TODO
    return bu, bi, P, Q
`,
        solution: `
import numpy as np

def predict(mu, bu, bi, P, Q, u, i):
    return float(mu + bu[u] + bi[i] + np.dot(P[u], Q[i]))

def sgd_epoch(ratings, mu, bu, bi, P, Q, lr, reg):
    bu, bi, P, Q = (np.array(a, dtype=float) for a in (bu, bi, P, Q))
    for u, i, r in ratings:
        e = r - predict(mu, bu, bi, P, Q, u, i)
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])
        pu, qi = P[u].copy(), Q[i].copy()
        P[u] = pu + lr * (e * qi - reg * pu)
        Q[i] = qi + lr * (e * pu - reg * qi)
    return bu, bi, P, Q
`,
        hints: ['Compute the error before changing anything.', 'Copy P[u] and Q[i] before updating so both updates use the old values.'],
        explain: `Each update nudges the user and item embeddings towards each other when the rating was higher than predicted, and apart when it was lower; the λ terms shrink everything slightly towards zero to fight overfitting.`,
        tests: [
          { name: 'the worked example', call: 'sgd_epoch([(0, 0, 5.0)], 3.0, np.zeros(1), np.zeros(1), np.array([[1.0, 0.0]]), np.array([[1.0, 1.0]]), 0.1, 0.0)', expect: '(np.array([0.1]), np.array([0.1]), np.array([[1.1, 0.1]]), np.array([[1.1, 1.0]]))' },
          { call: 'predict(3.0, np.array([0.5]), np.array([-0.2]), np.array([[1.0, 2.0]]), np.array([[0.5, 0.25]]), 0, 0)', expect: '4.3' },
          { name: 'one epoch on random data', setup: 'rng = np.random.default_rng(0)\nratings = [(int(rng.integers(6)), int(rng.integers(5)), float(rng.integers(1, 6))) for _ in range(30)]\nP0 = rng.normal(0, 0.1, (6, 2)); Q0 = rng.normal(0, 0.1, (5, 2))', call: 'sgd_epoch(ratings, 3.0, np.zeros(6), np.zeros(5), P0, Q0, 0.05, 0.02)', hidden: true },
          { name: 'inputs untouched and training error falls', setup: 'rng = np.random.default_rng(1)\nU, I = 20, 15\nTP, TQ = rng.normal(size=(U, 2)), rng.normal(size=(I, 2))\nratings = [(u, i, float(3 + TP[u] @ TQ[i])) for u in range(U) for i in range(I) if rng.random() < 0.5]\nmu = float(np.mean([r for _, _, r in ratings]))\nbu, bi = np.zeros(U), np.zeros(I)\nP, Q = rng.normal(0, 0.1, (U, 2)), rng.normal(0, 0.1, (I, 2))', code: 'P_before = P.copy()\ndef err(bu, bi, P, Q):\n    return np.mean([(r - predict(mu, bu, bi, P, Q, u, i)) ** 2 for u, i, r in ratings])\nstart = err(bu, bi, P, Q)\nstate = (bu, bi, P, Q)\nfor _ in range(40):\n    state = sgd_epoch(ratings, mu, *state, 0.05, 0.01)\nassert np.array_equal(P, P_before), "sgd_epoch must not modify its inputs"\nassert err(*state) < 0.5 * start', hidden: true }
        ]
      },
      {
        id: 'precision-at-k', title: 'Precision and Recall at k', difficulty: 'easy',
        prompt: `Implement:

- \`precision_at_k(recommended, relevant, k)\` → (number of the first k recommended items that are in \`relevant\`) / k
- \`recall_at_k(recommended, relevant, k)\` → the same count / len(relevant), or 0.0 if \`relevant\` is empty
- \`mean_at_k(metric, all_recommended, all_relevant, k)\` → the average of \`metric\` over all users

\`recommended\` is a ranked list; \`relevant\` is a set.

\`\`\`text
precision_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 3)  ->  1/3
recall_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 4)     ->  2/3
\`\`\``,
        starter: `
def precision_at_k(recommended, relevant, k):
    # TODO
    pass

def recall_at_k(recommended, relevant, k):
    # TODO
    pass

def mean_at_k(metric, all_recommended, all_relevant, k):
    # TODO
    pass
`,
        solution: `
def hits(recommended, relevant, k):
    return sum(1 for item in recommended[:k] if item in relevant)

def precision_at_k(recommended, relevant, k):
    return hits(recommended, relevant, k) / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return hits(recommended, relevant, k) / len(relevant)

def mean_at_k(metric, all_recommended, all_relevant, k):
    scores = [metric(rec, rel, k) for rec, rel in zip(all_recommended, all_relevant)]
    return sum(scores) / len(scores)
`,
        hints: ['Slice the ranked list with `recommended[:k]`.', 'Functions are values in Python: `metric(rec, rel, k)` calls whichever function was passed in.'],
        explain: `Precision@k asks "how much of what I showed was good?", recall@k asks "how much of the good stuff did I show?". Showing more items (larger k) usually raises recall and lowers precision.`,
        tests: [
          { call: 'precision_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 3)', expect: '1 / 3' },
          { call: 'recall_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 4)', expect: '2 / 3' },
          { name: 'averages over users', call: 'mean_at_k(precision_at_k, [["a", "b"], ["c", "d"]], [{"a"}, {"x"}], 2)', expect: '0.25' },
          { name: 'empty relevant set', call: 'recall_at_k(["a"], set(), 1)', expect: '0.0', hidden: true },
          { name: 'random recommendations vs reference', setup: 'rng = np.random.default_rng(0)\nrecs = [[int(x) for x in rng.permutation(20)[:10]] for _ in range(8)]\nrels = [set(int(x) for x in rng.choice(20, 4, replace=False)) for _ in range(8)]', call: '[mean_at_k(precision_at_k, recs, rels, 5), mean_at_k(recall_at_k, recs, rels, 10)]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Koren, Bell & Volinsky, Matrix Factorization Techniques for Recommender Systems', url: 'https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf' },
      { title: 'Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback', url: 'https://arxiv.org/abs/1205.2618' },
      { title: 'Covington et al., Deep Neural Networks for YouTube Recommendations', url: 'https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/' }
    ]
  },

  // =====================================================================================
  {
    id: 'mlops', branch: 'spec', title: 'MLOps: Deploying & Monitoring Models', short: 'MLOps', level: 2, minutes: 40, prereqs: ['evaluation-metrics', 'sklearn-workflow', 'sql-data-engineering'],
    tags: ['mlops', 'deployment', 'serving', 'monitoring', 'data drift', 'concept drift', 'psi', 'model registry', 'reproducibility', 'data validation', 'a/b testing'],
    summary: 'Getting models into production and keeping them healthy: the ML lifecycle, reproducibility and experiment tracking, batch versus online serving, testing and validating data, monitoring for data and concept drift with PSI, and safe rollouts with shadow, canary and A/B tests.',
    why: `Most ML projects fail not in the notebook but on the way to production, or quietly afterwards. Data changes, pipelines break, latency matters, and a model that scored 95% in testing slowly degrades without anyone noticing. **MLOps** applies software-engineering discipline (versioning, testing, automation, monitoring) to machine learning so that models stay correct and useful over time.`,
    objectives: [
      'Describe the ML lifecycle and what must be versioned for reproducibility',
      'Choose between batch and online serving, and package a model behind an API',
      'Validate incoming data and detect drift with the population stability index (PSI)',
      'Plan safe rollouts (shadow, canary, A/B) and retraining'
    ],
    sections: [
      { type: 'demo', title: 'The ML lifecycle', demo: 'dg-lifecycle', after: `To **reproduce** any model (to debug it, audit it or roll back to it) you must be able to recreate it exactly. Version all of:
- **code** (git), including feature-engineering code;
- **data** (snapshots or content hashes; tools such as DVC);
- **configuration and hyperparameters**, plus random seeds;
- **environment** (pinned dependency versions, a container image);
- **model artifacts**, in a **model registry** with stages such as "staging" and "production".

**Experiment tracking** (MLflow, Weights & Biases) records every training run's parameters, metrics and artifacts, so "which settings produced the good model last month?" has an answer.` },
      { type: 'text', title: 'Serving models', md: `| Mode | How | Good for |
| --- | --- | --- |
| **Batch** | score everything on a schedule and write results to a table | nightly churn scores, weekly recommendations: simple and cheap |
| **Online** | a web API answers each request in real time | fraud checks at payment time, search ranking: needs a latency budget |
| **Streaming / edge** | score events as they arrive, or on the device | sensors, keyboards, cameras: no network round trip |

Package the **whole pipeline**: preprocessing and model together, e.g. a scikit-learn \`Pipeline\` saved with joblib: load it once at start-up, validate every input, and return the model version with each prediction.

**Training–serving skew** is a classic silent bug: a feature computed one way in the training notebook and another way in production. Share the feature code between the two, or use a feature store.` },
      { type: 'code', title: 'Serving a scikit-learn pipeline with FastAPI', run: false, md: `Run locally: \`pip install fastapi uvicorn scikit-learn pandas joblib\`, save a fitted pipeline with \`joblib.dump(pipeline, "model-v3.joblib")\`, then start the server with \`uvicorn app:app --port 8000\`.`, code: `
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_VERSION = "v3"
model = joblib.load("model-v3.joblib")          # the whole Pipeline: preprocessing + model

class House(BaseModel):                           # input validation for free
    sqm: float = Field(gt=10, lt=2000)
    rooms: int = Field(ge=1, le=20)
    city: str

app = FastAPI()

@app.post("/predict")
def predict(house: House):
    X = pd.DataFrame([house.model_dump()])
    price = float(model.predict(X)[0])
    return {"price": price, "model_version": MODEL_VERSION}
` },
      { type: 'text', title: 'Testing ML systems', md: `- **Unit tests** for feature code and data transformations.
- **Data validation** on every batch: schema, types, allowed ranges, missing values, category sets (tools: Great Expectations, pandera). Reject or quarantine bad data instead of silently predicting on it.
- **Model tests**: a minimum score on a fixed evaluation set; *invariance* tests (changing a customer's name must not change their credit score); *directional* tests (a bigger house should not lower the predicted price).
- **CI/CD**: on every change, run the tests, retrain or re-evaluate, and compare against the current production model before promoting anything.` },
      { type: 'text', title: 'Monitoring and drift', md: `The world keeps changing after deployment:

- **Data drift** (covariate shift): the distribution of the inputs changes (new customer segments, a recalibrated sensor, a new app version).
- **Concept drift**: the relationship between inputs and outputs changes (fraudsters adapt, tastes change, a pandemic rewrites shopping habits).
- **Label delay**: true outcomes (did the loan default?) can arrive months later, so monitor the **inputs and the predictions** as early-warning signals.

The **population stability index** compares a feature's distribution in production with the training data, bin by bin:

$$ PSI = Σ_bins ( a_i − e_i ) · ln( a_i / e_i ) $$
:::where
e_i — the share of training (expected) values that fall into bin i
a_i — the share of production (actual) values in the same bin
ln — the natural logarithm
Σ_bins — add up over the bins (often 10 bins at the training data's deciles)
A common rule of thumb reads PSI below 0.1 as stable, 0.1 to 0.25 as a moderate shift and above 0.25 as a significant shift.
:::

Also track operational health: latency, error rates, throughput and cost.` },
      { type: 'code', title: 'Detecting drift with PSI', code: `
import numpy as np
rng = np.random.default_rng(0)

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1]       # inner edges from training deciles
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=bins) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=bins) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)                  # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))

train_income = rng.lognormal(10.5, 0.5, 20_000)
scenarios = [("same distribution", rng.lognormal(10.5, 0.5, 5_000)),
             ("incomes up 10%", rng.lognormal(10.5, 0.5, 5_000) * 1.10),
             ("incomes up 25%", rng.lognormal(10.5, 0.5, 5_000) * 1.25),
             ("new customer segment", np.concatenate([rng.lognormal(10.5, 0.5, 3_000), rng.lognormal(9.5, 0.4, 2_000)]))]
for label, production in scenarios:
    value = psi(train_income, production)
    level = "stable" if value < 0.1 else "moderate shift" if value < 0.25 else "significant shift"
    print(f"{label:22s} PSI = {value:.3f}  -> {level}")
` },
      { type: 'chart', title: "PSI as an input distribution shifts", md: `A feature whose distribution shifts by a growing amount, compared with its training-time reference (10 quantile bins, 5,000 new values):`, after: `PSI grows roughly with the square of the shift. By the usual rules of thumb, a shift of about 0.4 standard deviations crosses the "watch" line (PSI ≈ 0.14) and one of about half a standard deviation reaches the "investigate" line (PSI ≈ 0.24). The thresholds are conventions, not laws: tune them per feature, and remember that drift in an unimportant feature may not hurt the model at all, while a small drift in a key feature might.`, chart: {"kind":"line","x":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],"series":[{"name":"PSI of the shifted data against the reference","y":[0.003,0.008,0.043,0.072,0.143,0.238,0.333,0.473,0.635,0.715,0.905]}],"xLabel":"shift of the mean, in standard deviations","yLabel":"population stability index","hlines":[{"y":0.1,"label":"0.1: watch"},{"y":0.25,"label":"0.25: investigate"}],"yMin":0} },
      { type: 'check', question: { type: 'mcq', q: 'Fraudsters change their tactics, so the same transaction features now mean something different. What is this?', options: ['Data drift', 'Concept drift', 'Label leakage', 'Underfitting'], answer: 1, explain: 'The relationship P(y | x) changed, not just the input distribution.' } },
      { type: 'text', title: 'Safe rollouts and retraining', md: `- **Shadow deployment**: the new model scores live traffic, but its predictions are only logged, not used. You compare it with the current model at zero risk.
- **Canary release**: send a small share of traffic (say 5%) to the new model, watch the metrics, then ramp up.
- **A/B test**: split users randomly and compare business metrics with proper statistics.
- **Rollback**: always keep the previous model ready to switch back to instantly.

Retrain on a schedule (e.g. weekly) or when monitoring triggers it, but **always evaluate before promoting**: a model retrained on bad or drifted data can be worse than the old one.` },
      { type: 'text', title: "Reproducibility, lineage and feature stores", md: `Months after deployment someone will ask which data and code produced a model. Answering requires recording, for every model version:
- the **code** version (a git commit) and the environment (library versions, container image);
- the **data** version: a snapshot or versioned table (tools such as DVC or lakeFS), or the exact partitions used;
- the **hyperparameters**, metrics and evaluation reports, kept in an experiment tracker (MLflow, Weights & Biases);
- the **model artefact** itself, registered with a version and a stage (staging, production, archived) in a model registry.

The most common production bug is **training–serving skew**: a feature computed one way in the training pipeline and slightly differently in the serving code (another time zone, a missing default, an outdated lookup table). The model then receives inputs it never saw in training, and nothing crashes. **Feature stores** address this by defining each feature once and serving the same computation both to training, with point-in-time correct history (see [SQL & Data Engineering for ML](#/concept/sql-data-engineering)), and to real-time inference.` }
    ],
    when: `- As soon as a model's predictions are used by people or systems, not just in a notebook.
- Batch serving whenever predictions can be precomputed; online serving only when inputs are only known at request time.
- Drift monitoring for every feature that matters, with alerts tied to retraining or investigation.`,
    pitfalls: `- No record of which data, code and parameters produced the production model.
- Training–serving skew from feature code that differs between training and production.
- Monitoring only infrastructure (latency, errors) and not the data and predictions.
- Retraining automatically without evaluation gates.
- Deploying without a rollback plan.`,
    takeaways: [
      'Version code, data, configuration, environment and models so any result can be reproduced.',
      'Serve the whole pipeline; prefer batch when possible; validate inputs.',
      'Test ML systems: data validation, invariance and directional tests, evaluation gates in CI.',
      'Monitor for data drift (PSI) and concept drift; labels often arrive late.',
      'Roll out with shadow, canary and A/B tests, and keep a rollback ready.'
    ],
    quiz: [
      { type: 'mcq', q: 'What is training–serving skew?', options: ['Using a bigger model in production', 'Features computed differently in training and in production, causing silent errors', 'Training on more data than you serve', 'Serving predictions too slowly'], answer: 1, explain: 'The model sees inputs in production that don\'t match what it learned from.' },
      { type: 'fill', q: 'A feature\'s PSI between training and production is 0.31. Using the common rule of thumb, is the shift stable, moderate or significant?', answer: ['significant'], explain: 'Above 0.25 counts as a significant shift.' },
      { type: 'mcq', q: 'What is a shadow deployment?', options: ['Deploying at night', 'Running the new model on live traffic while only logging its predictions, without using them', 'Deleting the old model', 'Serving only to internal users'], answer: 1, explain: 'It lets you compare models with zero user impact.' },
      { type: 'tf', q: 'If production inputs look exactly like the training data, the model cannot be degrading.', answer: false, explain: 'Concept drift changes the input–output relationship even when the inputs look the same.' },
      { type: 'mcq', q: 'Which item is NOT needed to reproduce a trained model?', options: ['The code version', 'The data snapshot', 'The random seeds and hyperparameters', 'The colour theme of the training notebook'], answer: 3, explain: 'Everything that affects the computation must be versioned; cosmetics need not be.' },
      {"type":"mcq","q":"A feature is computed with UTC timestamps in training but with local time in the serving code. What is this called?","options":["Concept drift","Training–serving skew","Overfitting","Label leakage"],"answer":1,"explain":"The same feature is computed differently in the two pipelines, so the model sees inputs at serving time that differ from what it was trained on."},
      {"type":"tf","q":"A PSI below 0.1 guarantees that the model's accuracy has not changed.","answer":false,"explain":"PSI only measures how one input's distribution moved. The relationship between inputs and outcomes can change (concept drift) without any change in the inputs."}
    ],
    problems: [
      {
        id: 'psi-drift', title: 'Population Stability Index', difficulty: 'medium',
        prompt: `Implement \`psi(expected, actual, bins=10)\` and \`drift_level(value)\`:

1. Inner bin edges: \`np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1]\` (the first and last bins are open-ended).
2. Bin index of each value: \`np.searchsorted(edges, values, side="right")\`; counts per bin with \`np.bincount(..., minlength=bins)\`; divide by the number of values to get shares e (training) and a (production).
3. Clip both share arrays to at least 1e-6, then return PSI = Σ (a − e) · ln(a / e) as a float.

\`drift_level(value)\` returns "stable" if value < 0.1, "moderate" if value < 0.25, otherwise "significant".

\`\`\`text
psi(x, x)  ->  0.0   for any array x
\`\`\``,
        starter: `
import numpy as np

def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    # TODO
    pass

def drift_level(value):
    # TODO
    pass
`,
        solution: `
import numpy as np

def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1]
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=bins) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=bins) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def drift_level(value):
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"
`,
        hints: ['`np.searchsorted(edges, values, side="right")` gives each value\'s bin number from 0 to bins − 1.', '`np.bincount(idx, minlength=bins)` counts values per bin.'],
        explain: `Each term (a − e) · ln(a / e) is never negative, so PSI is 0 only when the two distributions put the same share in every bin. It is a symmetric form of the KL divergence, computed on binned data.`,
        tests: [
          { name: 'identical data', setup: 'rng = np.random.default_rng(0)\nx = rng.normal(size=1000)', call: 'psi(x, x)', expect: '0.0' },
          { call: '[drift_level(0.05), drift_level(0.1), drift_level(0.3)]', expect: '["stable", "moderate", "significant"]' },
          { name: 'a shifted distribution', setup: 'rng = np.random.default_rng(1)\ntrain = rng.normal(0, 1, 5000)\nprod = rng.normal(0.8, 1, 2000)', call: 'psi(train, prod)' },
          { name: 'large shift is significant', setup: 'rng = np.random.default_rng(2)\ntrain = rng.normal(0, 1, 5000)\nprod = rng.normal(1.5, 1, 2000)', code: 'assert drift_level(psi(train, prod)) == "significant"', hidden: true },
          { name: 'other bin counts', setup: 'rng = np.random.default_rng(3)\ntrain = rng.exponential(1.0, 3000)\nprod = rng.exponential(1.3, 1000)', call: '[psi(train, prod, 5), psi(train, prod, 20)]', hidden: true }
        ]
      },
      {
        id: 'validate-rows', title: 'Validate Incoming Data', difficulty: 'easy',
        prompt: `Before scoring a batch, check it against a schema. Implement \`validate_rows(rows, schema)\` → a list of error strings (empty if all rows are valid). \`rows\` is a list of dicts; \`schema\` maps a column name to \`(type, min, max)\`, where min and max may be \`None\`.

For each row index r, check the schema columns **in schema order**, reporting at most one error per column:
- missing → \`"row R: missing COL"\`
- wrong type → \`"row R: COL has wrong type"\` (\`int\` rejects booleans; \`float\` accepts ints but not booleans)
- below min or above max → \`"row R: COL out of range"\`

Then, for each key of the row that is not in the schema (in the row's key order): \`"row R: unexpected column COL"\`.

\`\`\`text
schema = {"age": (int, 18, 120), "country": (str, None, None)}
validate_rows([{"age": 30, "country": "NL"}, {"age": 12}], schema)
-> ["row 1: age out of range", "row 1: missing country"]
\`\`\``,
        starter: `
def validate_rows(rows, schema):
    errors = []
    # TODO
    return errors
`,
        solution: `
def type_ok(value, expected):
    if isinstance(value, bool) and expected is not bool:
        return False
    if expected is float:
        return isinstance(value, (int, float))
    return isinstance(value, expected)

def validate_rows(rows, schema):
    errors = []
    for r, row in enumerate(rows):
        for col, (typ, lo, hi) in schema.items():
            if col not in row:
                errors.append(f"row {r}: missing {col}")
            elif not type_ok(row[col], typ):
                errors.append(f"row {r}: {col} has wrong type")
            elif (lo is not None and row[col] < lo) or (hi is not None and row[col] > hi):
                errors.append(f"row {r}: {col} out of range")
        for col in row:
            if col not in schema:
                errors.append(f"row {r}: unexpected column {col}")
    return errors
`,
        hints: ['Use `enumerate(rows)` to get each row\'s index.', 'An if/elif chain naturally reports at most one error per column.', 'Check booleans before ints: `isinstance(True, int)` is True.'],
        explain: `Validating inputs at the boundary turns silent garbage predictions into loud, actionable errors. Libraries like pandera or Great Expectations do the same with richer checks (regexes, uniqueness, distributions).`,
        tests: [
          { setup: 'schema = {"age": (int, 18, 120), "country": (str, None, None)}', call: 'validate_rows([{"age": 30, "country": "NL"}, {"age": 12}], schema)', expect: '["row 1: age out of range", "row 1: missing country"]' },
          { name: 'all valid', setup: 'schema = {"income": (float, 0, None)}', call: 'validate_rows([{"income": 52000}, {"income": 1.5}], schema)', expect: '[]' },
          { name: 'types and extra columns', setup: 'schema = {"age": (int, 0, None), "score": (float, 0.0, 1.0)}', call: 'validate_rows([{"age": True, "score": "0.5", "id": 7}, {"age": 5, "score": 1.5}], schema)', expect: '["row 0: age has wrong type", "row 0: score has wrong type", "row 0: unexpected column id", "row 1: score out of range"]', hidden: true },
          { name: 'random batch vs reference', setup: 'rng = np.random.default_rng(0)\nschema = {"age": (int, 18, 99), "ok": (bool, None, None)}\nrows = []\nfor _ in range(12):\n    row = {}\n    if rng.random() < 0.9:\n        row["age"] = int(rng.integers(10, 110))\n    if rng.random() < 0.8:\n        row["ok"] = bool(rng.random() < 0.5) if rng.random() < 0.8 else 1\n    if rng.random() < 0.2:\n        row["extra"] = 0\n    rows.append(row)', call: 'validate_rows(rows, schema)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Sculley et al., Hidden Technical Debt in Machine Learning Systems', url: 'https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems' },
      { title: 'Breck et al., The ML Test Score: A Rubric for ML Production Readiness', url: 'https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/' },
      { title: 'Chip Huyen, Designing Machine Learning Systems', url: 'https://huyenchip.com/books/' },
      { title: 'MLflow documentation', url: 'https://mlflow.org/docs/latest/index.html' }
    ]
  },

  // =====================================================================================
  {
    id: 'responsible-ai', branch: 'spec', title: 'Responsible AI: Fairness, Explainability & Safety', short: 'Responsible AI', level: 2, minutes: 40, prereqs: ['evaluation-metrics'],
    tags: ['responsible ai', 'fairness', 'bias', 'demographic parity', 'equal opportunity', 'equalized odds', 'explainability', 'permutation importance', 'shap', 'privacy', 'model cards', 'ai safety'],
    summary: 'Building ML that is fair, transparent, private and safe: where bias comes from, group fairness metrics (demographic parity, equal opportunity, equalized odds) and their trade-offs, explainability with permutation importance and SHAP, privacy, documentation with model cards, and AI-safety basics.',
    why: `ML systems now help decide who gets a loan, a job interview, medical attention or extra police scrutiny, and they can silently reproduce or amplify historical bias, fail for under-represented groups, leak private data or be misused. Responsible AI is not a final checkbox: it means measuring these risks like any other metric, explaining decisions, documenting limitations, and complying with growing regulation such as the EU AI Act.`,
    objectives: [
      'Identify where bias enters the ML lifecycle',
      'Compute and interpret demographic parity, equal opportunity and equalized-odds gaps',
      'Explain a model\'s behaviour with permutation importance, and describe SHAP',
      'Apply privacy, documentation (model cards) and safety practices'
    ],
    sections: [
      { type: 'text', title: 'Where bias comes from', md: `- **Historical bias**: the data faithfully records past discrimination (e.g. who was hired or promoted).
- **Representation bias**: some groups are under-represented in the data (e.g. face datasets dominated by lighter skin).
- **Measurement bias**: the label is a skewed proxy for what you care about (arrests as a proxy for crime; past healthcare spending as a proxy for medical need).
- **Aggregation bias**: one model for groups whose patterns genuinely differ.
- **Evaluation bias**: test sets that don't represent the people the system will serve.
- **Feedback loops**: the model's decisions shape the next round of data.

Simply deleting the sensitive attribute ("fairness through unawareness") rarely works: postcode, name, school or shopping habits can act as **proxies** for it.` },
      { type: 'text', title: 'Group fairness metrics', md: `For a binary decision ŷ (1 = approve), true outcome y and group g:

$$ selection rate_g = P(ŷ = 1 | g)          TPR_g = P(ŷ = 1 | y = 1, g)          FPR_g = P(ŷ = 1 | y = 0, g) $$
:::where
ŷ — the model's decision (1 = positive, e.g. approve the loan)
y — the true outcome (1 = the person would in fact repay)
g — the group a person belongs to (e.g. by sex or ethnicity)
P(… | …) — a probability computed only among the people matching the condition after the bar
selection rate_g — how often group g receives the positive decision
TPR_g — the true-positive rate: how often qualified members of g are approved
FPR_g — the false-positive rate: how often unqualified members of g are approved
:::

- **Demographic parity**: equal selection rates across groups.
- **Equal opportunity**: equal TPR (qualified people are approved equally often whatever their group).
- **Equalized odds**: equal TPR *and* equal FPR.
- The **disparate impact ratio** (lowest selection rate ÷ highest) is often checked against the "four-fifths rule" (below 0.8 is a red flag).

These criteria can conflict: when base rates differ between groups, a classifier generally cannot satisfy calibration and equalized odds at the same time. Choosing a criterion is a value judgment about which errors cause harm: involve domain experts, affected communities and legal review. Mitigations act before training (re-weighting data), during training (fairness constraints) or after it (group-specific thresholds).` },
      { type: 'chart', title: "Selection rates of two groups under one threshold", md: `Model scores whose distribution differs between two groups, approved whenever the score passes a single threshold:`, after: `At any single threshold the two groups are approved at different rates: at 0.5, 73% of group A against 42% of group B. Whether that gap is unfair depends on why the scores differ, which is exactly what the fairness criteria disagree about: demographic parity would demand equal approval rates (different thresholds per group), while equal opportunity looks only at people who would have repaid.`, chart: {"kind":"line","x":[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8],"series":[{"name":"group A approved","y":[0.977,0.958,0.93,0.893,0.846,0.791,0.726,0.655,0.575,0.491,0.405,0.315,0.219]},{"name":"group B approved","y":[0.88,0.812,0.743,0.662,0.582,0.5,0.423,0.345,0.272,0.206,0.148,0.098,0.057]}],"xLabel":"approval threshold on the model's score","yLabel":"share approved","yMin":0,"yMax":1,"vlines":[{"x":0.5,"label":"a typical threshold"}]} },
      { type: 'code', title: 'Auditing a loan model for fairness', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 4000
group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
qualified = rng.random(n) < np.where(group == "A", 0.5, 0.4)        # would actually repay
# the score is informative, but a biased proxy feature lowers it for group B
score = rng.normal(np.where(qualified, 0.65, 0.35), 0.15) - np.where(group == "B", 0.08, 0.0)

def audit(threshold_a, threshold_b):
    approve = score >= np.where(group == "A", threshold_a, threshold_b)
    for g in ["A", "B"]:
        m = group == g
        print(f"  group {g}: selection rate {approve[m].mean():.2f}   TPR {approve[m & qualified].mean():.2f}   FPR {approve[m & ~qualified].mean():.2f}")

print("one threshold (0.50) for everyone:")
audit(0.50, 0.50)
print("threshold 0.42 for group B, chosen to equalize TPR (equal opportunity):")
audit(0.50, 0.42)
`, after: `With one threshold, qualified people in group B are approved far less often (TPR 0.69 versus 0.85) because of the biased score. Adjusting B's threshold equalizes both TPR and FPR here. Real decisions about such adjustments also involve law and ethics, not only arithmetic.` },
      { type: 'check', question: { type: 'mcq', q: 'Equal opportunity requires equal values of which metric across groups?', options: ['Selection rate', 'True-positive rate', 'Accuracy', 'Training loss'], answer: 1, explain: 'Qualified people should be approved at the same rate in every group.' } },
      { type: 'text', title: "Why the fairness criteria conflict", md: `It would be convenient to satisfy every fairness criterion at once. When the groups have different base rates (different shares of positive outcomes), that is mathematically impossible except for perfect or trivial classifiers. For a binary classifier, each group's error rates are tied together by its base rate:

$$ FPR = p / (1 − p) · (1 − PPV) / PPV · (1 − FNR) $$
:::where
p — the group's base rate: the share of people in the group whose true outcome is positive
PPV — the positive predictive value: the share of positive predictions that are correct (equal PPV across groups is one form of calibration)
FNR — the false negative rate: the share of true positives the classifier misses
FPR — the false positive rate: the share of true negatives that are wrongly flagged
:::

If two groups have different values of p, then equal PPV and equal FNR force different FPRs, and similarly for every other pair of criteria (results by Chouldechova, and by Kleinberg and colleagues, in 2016). Fairness therefore always involves a choice about which errors matter most in the application (a good borrower wrongly refused a loan, or an innocent customer wrongly flagged for fraud), and that choice should be made explicitly, with the people affected in mind, not left to whatever a default threshold happens to do.` },
      { type: 'text', title: 'Explainability', md: `Explanations can be **global** (which features matter overall?) or **local** (why did *this* applicant get rejected?). Simple models (linear models, small trees) are interpretable by design; complex ones need post-hoc tools.

**Permutation importance** (global, model-agnostic): shuffle one feature's column in the validation data, breaking its link to the target, and measure how much the score drops:

$$ importance_j = score(X) − score(X with column j shuffled) $$
:::where
X — the validation features
score — any metric where higher is better, such as accuracy or R²
column j shuffled — feature j's values randomly permuted across rows, so they no longer match their rows
importance_j — how much the model relies on feature j; about 0 means it doesn't use it
:::

**SHAP** (local) uses Shapley values from game theory to split one prediction into contributions from each feature: the contributions add up to the prediction minus the average prediction. **Partial dependence plots** show how the prediction changes as one feature varies.

Caveats: correlated features share or "steal" importance from each other, and explanations describe the *model*, not causal truths about the world.` },
      { type: 'code', title: 'Permutation importance from scratch', code: `
import numpy as np
rng = np.random.default_rng(0)
n = 1000
X = rng.normal(size=(n, 5))
y = 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + rng.normal(0, 0.5, n)     # features 3 and 4 are pure noise
names = ["size", "rooms", "distance", "noise_1", "noise_2"]
Xtr, Xva, ytr, yva = X[:800], X[800:], y[:800], y[800:]

w = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], ytr, rcond=None)[0]     # a linear regression model
predict = lambda X: np.c_[X, np.ones(len(X))] @ w
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

base = r2(yva, predict(Xva))
print(f"validation R2 = {base:.3f}")
for j, name in enumerate(names):
    drops = []
    for _ in range(5):
        Xp = Xva.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])              # break the link between feature j and y
        drops.append(base - r2(yva, predict(Xp)))
    importance = round(float(np.mean(drops)), 3) + 0.0     # + 0.0 turns -0.0 into 0.0 for printing
    print(f"{name:9s} importance {importance:.3f} ± {np.std(drops):.3f}")
`, after: `The ranking matches how the data was generated: size matters most, then distance, then rooms, and the two noise features score about zero.` },
      { type: 'text', title: 'Privacy, documentation and safety', md: `- **Privacy**: collect only the data you need, protect and anonymize it, and remember that models can memorize and leak training examples (LLMs can regurgitate text; membership-inference attacks can reveal whether someone was in the training data). **Differential privacy** adds calibrated noise during training to bound what can be learned about any individual; **federated learning** trains on devices without centralizing raw data.
- **Documentation**: **model cards** state the intended use, performance broken down by group, limitations and ethical considerations; **datasheets for datasets** record how data was collected and labelled.
- **Safety and misuse**: test robustness (adversarial examples), red-team the system, keep humans in the loop for high-stakes decisions, and give people a way to contest decisions. For LLMs this includes harmful content, hallucinations, jailbreaks and prompt injection.
- **Regulation**: the EU AI Act sorts systems into risk tiers (unacceptable, high, limited, minimal); high-risk uses such as hiring and credit scoring require risk management, data governance, documentation, transparency and human oversight.` },
      { type: 'check', question: { type: 'tf', q: 'Removing the sensitive attribute (e.g. ethnicity) from the features guarantees a fair model.', answer: false, explain: 'Other features can act as proxies for it.' } }
    ],
    when: `- Always for systems that affect people, and especially in high-stakes domains: credit, hiring, housing, healthcare, education, justice.
- Fairness audits by subgroup before launch and continuously after; explanations for users who are affected and for debugging.
- Model cards and datasheets for every model you ship or share.`,
    pitfalls: `- Reporting only overall accuracy, which hides poor performance for small groups.
- Assuming that removing a sensitive attribute removes bias.
- Picking a fairness metric without thinking about which errors cause harm.
- Treating feature importance as causation.
- Shipping without documentation, monitoring or a way for people to appeal.`,
    takeaways: [
      'Bias enters through data, labels, aggregation, evaluation and feedback loops; proxies defeat "unawareness".',
      'Demographic parity = equal selection rates; equal opportunity = equal TPR; equalized odds = equal TPR and FPR.',
      'Fairness criteria can conflict; choosing one is a value judgment that needs stakeholders.',
      'Permutation importance and SHAP explain models (not the world); correlated features complicate them.',
      'Protect privacy, document with model cards, red-team, keep humans in the loop, and follow regulation.'
    ],
    quiz: [
      { type: 'mcq', q: 'A hiring model selects 30% of group A and 15% of group B. Which criterion is violated?', options: ['Equal opportunity, necessarily', 'Demographic parity', 'Calibration', 'None'], answer: 1, explain: 'Selection rates differ; whether TPRs differ needs the true outcomes.' },
      { type: 'fill', q: 'Selection rates are 0.40 for group A and 0.28 for group B. What is the disparate impact ratio? (2 decimals)', answer: ['0.7', '0.70'], explain: '0.28 / 0.40 = 0.70, below the 0.8 "four-fifths" threshold.' },
      { type: 'mcq', q: 'How is permutation importance computed?', options: ['By removing the feature and retraining', 'By shuffling the feature\'s values in validation data and measuring how much the score drops', 'By reading the model\'s weights', 'By counting how often a feature is used'], answer: 1, explain: 'Shuffling breaks the feature\'s link with the target without retraining.' },
      { type: 'tf', q: 'SHAP values for one prediction add up to the difference between that prediction and the average prediction.', answer: true, explain: 'This additivity is a key property of Shapley values.' },
      { type: 'mcq', q: 'What does a model card document?', options: ['The model\'s source code', 'Intended use, performance by group, limitations and ethical considerations', 'The GPU used for training', 'The training loss curve only'], answer: 1, explain: 'It helps others use the model responsibly.' },
      {"type":"mcq","q":"Two groups have different base rates. Which statement is true for an imperfect classifier?","options":["It can always satisfy calibration, equal false positive rates and equal false negative rates at once","It cannot satisfy all three at once, so some fairness criteria must be traded off","Fairness cannot be measured","Base rates do not affect error rates"],"answer":1,"explain":"The identity FPR = p/(1 − p) · (1 − PPV)/PPV · (1 − FNR) ties the three together through the base rate p, so different base rates force at least one of them to differ."},
      {"type":"fill","q":"In a group with base rate p = 0.5, a classifier has PPV = 0.8 and FNR = 0.2. What is its false positive rate? (decimal)","answer":["0.2",".2"],"explain":"FPR = p/(1 − p) × (1 − PPV)/PPV × (1 − FNR) = 1 × 0.25 × 0.8 = 0.2."}
    ],
    problems: [
      {
        id: 'fairness-metrics', title: 'Fairness Metrics by Group', difficulty: 'medium',
        prompt: `Implement:

- \`group_metrics(y_true, y_pred, groups)\` → a dict mapping each group to \`{"selection_rate": ..., "tpr": ..., "fpr": ...}\` (Python floats). All three inputs are equal-length arrays or lists; you may assume every group has both positive and negative true labels.
- \`fairness_report(y_true, y_pred, groups)\` → \`{"dp_diff": ..., "eo_diff": ..., "fpr_diff": ..., "di_ratio": ...}\`: the largest minus the smallest selection rate, TPR and FPR across groups, and the disparate impact ratio (smallest selection rate ÷ largest).

\`\`\`text
y_true = [1, 1, 0, 0, 1, 1, 0, 0]
y_pred = [1, 1, 1, 0, 1, 0, 0, 0]
groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
group_metrics(...)  ->  {"A": {"selection_rate": 0.75, "tpr": 1.0, "fpr": 0.5},
                         "B": {"selection_rate": 0.25, "tpr": 0.5, "fpr": 0.0}}
\`\`\``,
        starter: `
import numpy as np

def group_metrics(y_true, y_pred, groups):
    # TODO
    pass

def fairness_report(y_true, y_pred, groups):
    # TODO
    pass
`,
        solution: `
import numpy as np

def group_metrics(y_true, y_pred, groups):
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = {
            "selection_rate": float(np.mean(y_pred[m] == 1)),
            "tpr": float(np.mean(y_pred[m & (y_true == 1)] == 1)),
            "fpr": float(np.mean(y_pred[m & (y_true == 0)] == 1)),
        }
    return out

def fairness_report(y_true, y_pred, groups):
    gm = group_metrics(y_true, y_pred, groups)
    sel = [v["selection_rate"] for v in gm.values()]
    tpr = [v["tpr"] for v in gm.values()]
    fpr = [v["fpr"] for v in gm.values()]
    return {"dp_diff": max(sel) - min(sel), "eo_diff": max(tpr) - min(tpr),
            "fpr_diff": max(fpr) - min(fpr), "di_ratio": min(sel) / max(sel)}
`,
        hints: ['Boolean masks combine with `&`: `(groups == g) & (y_true == 1)`.', 'The mean of a boolean array is the fraction of True values.', '`np.unique(groups)` lists each group once.'],
        explain: `A fairness audit is just ordinary metrics computed per group, but deciding which gap matters, and what to do about it, requires understanding the harms of each kind of error in the real application.`,
        tests: [
          { setup: 'y_true = [1, 1, 0, 0, 1, 1, 0, 0]\ny_pred = [1, 1, 1, 0, 1, 0, 0, 0]\ngroups = ["A", "A", "A", "A", "B", "B", "B", "B"]', call: 'group_metrics(y_true, y_pred, groups)', expect: '{"A": {"selection_rate": 0.75, "tpr": 1.0, "fpr": 0.5}, "B": {"selection_rate": 0.25, "tpr": 0.5, "fpr": 0.0}}' },
          { setup: 'y_true = [1, 1, 0, 0, 1, 1, 0, 0]\ny_pred = [1, 1, 1, 0, 1, 0, 0, 0]\ngroups = ["A", "A", "A", "A", "B", "B", "B", "B"]', call: 'fairness_report(y_true, y_pred, groups)', expect: '{"dp_diff": 0.5, "eo_diff": 0.5, "fpr_diff": 0.5, "di_ratio": 1 / 3}' },
          { name: 'three groups vs reference', setup: 'rng = np.random.default_rng(0)\nn = 300\ngroups = rng.choice(["x", "y", "z"], n)\ny_true = (rng.random(n) < 0.5).astype(int)\ny_pred = (rng.random(n) < np.where(groups == "z", 0.3, 0.6)).astype(int)', call: 'fairness_report(y_true, y_pred, groups)', hidden: true },
          { name: 'a perfectly fair classifier', setup: 'y_true = [1, 0, 1, 0]\ny_pred = [1, 0, 1, 0]\ngroups = ["a", "a", "b", "b"]', call: 'fairness_report(y_true, y_pred, groups)', expect: '{"dp_diff": 0.0, "eo_diff": 0.0, "fpr_diff": 0.0, "di_ratio": 1.0}', hidden: true }
        ]
      },
      {
        id: 'permutation-importance', title: 'Permutation Importance', difficulty: 'medium',
        prompt: `Implement \`permutation_importance(predict, X, y, metric, n_repeats=5, seed=0)\` → a NumPy array with one importance per feature:

- \`base = metric(y, predict(X))\`; \`rng = np.random.default_rng(seed)\`.
- For each feature j (in order 0, 1, …) and each repeat (in order): copy X, replace column j with \`rng.permutation(X[:, j])\`, and record \`base − metric(y, predict(X_perm))\`.
- The importance of feature j is the mean of its recorded drops.

Use the random generator exactly in this order so your results match the tests. \`X\` itself must not be modified.`,
        starter: `
import numpy as np

def permutation_importance(predict, X, y, metric, n_repeats=5, seed=0):
    X = np.asarray(X, dtype=float)
    # TODO
    pass
`,
        solution: `
import numpy as np

def permutation_importance(predict, X, y, metric, n_repeats=5, seed=0):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    base = metric(y, predict(X))
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(X[:, j])
            drops.append(base - metric(y, predict(Xp)))
        importances[j] = np.mean(drops)
    return importances
`,
        hints: ['Compute the baseline score once.', '`X.copy()` before shuffling keeps the original data intact.', '`rng.permutation(column)` returns a shuffled copy.'],
        explain: `Because it only needs predictions, permutation importance works for any model: a neural network, a gradient-boosted ensemble or a black-box API. Repeating the shuffle and averaging reduces the randomness of a single permutation.`,
        tests: [
          { name: 'a model that only uses feature 0', setup: 'rng = np.random.default_rng(0)\nX = rng.normal(size=(200, 3))\ny = 2 * X[:, 0]\npredict = lambda X: 2 * X[:, 0]\nmse_score = lambda y, p: -np.mean((y - p) ** 2)', code: 'imp = permutation_importance(predict, X, y, mse_score)\nassert imp.shape == (3,)\nassert imp[0] > 1.0 and abs(imp[1]) < 1e-12 and abs(imp[2]) < 1e-12, imp' },
          { name: 'matches the reference', setup: 'rng = np.random.default_rng(1)\nX = rng.normal(size=(100, 4))\nw = np.array([1.0, -2.0, 0.5, 0.0])\ny = X @ w + rng.normal(0, 0.1, 100)\npredict = lambda X: X @ w\nr2 = lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)', call: 'permutation_importance(predict, X, y, r2, n_repeats=3, seed=4)' },
          { name: 'X is not modified', setup: 'rng = np.random.default_rng(2)\nX = rng.normal(size=(50, 2))\ny = X[:, 1]\npredict = lambda X: X[:, 1]\nscore = lambda y, p: -np.mean(np.abs(y - p))', code: 'X0 = X.copy()\npermutation_importance(predict, X, y, score)\nassert np.array_equal(X, X0)', hidden: true },
          { name: 'a classifier with accuracy', setup: 'rng = np.random.default_rng(3)\nX = rng.normal(size=(300, 3))\ny = (X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)\npredict = lambda X: (X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)\nacc = lambda y, p: np.mean(y == p)', call: 'permutation_importance(predict, X, y, acc, n_repeats=4, seed=1)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Barocas, Hardt & Narayanan, Fairness and Machine Learning (free book)', url: 'https://fairmlbook.org/' },
      { title: 'Mitchell et al., Model Cards for Model Reporting', url: 'https://arxiv.org/abs/1810.03993' },
      { title: 'Lundberg & Lee, A Unified Approach to Interpreting Model Predictions (SHAP)', url: 'https://arxiv.org/abs/1705.07874' },
      { title: 'Molnar, Interpretable Machine Learning (free book)', url: 'https://christophm.github.io/interpretable-ml-book/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
