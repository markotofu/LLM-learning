/* ML Atlas – content: AI governance, contextual bandits, geometric deep learning, interpretability in practice, AI hardware */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'ai-governance', branch: 'spec', title: 'AI Governance & Regulation', short: 'AI Governance', level: 2, minutes: 40, prereqs: ['responsible-ai'],
    tags: ['ai governance', 'regulation', 'eu ai act', 'high-risk ai', 'general-purpose ai', 'systemic risk', 'nist ai rmf', 'iso 42001', 'model card', 'impact assessment', 'audit', 'compliance', 'policy'],
    summary: 'The rules and practices that shape how AI is built and deployed: the EU AI Act and its risk tiers, obligations for general-purpose models and the 10²⁵ FLOP threshold, the patchwork of US federal guidance and state laws, approaches in the UK and China, international standards such as the NIST AI Risk Management Framework and ISO/IEC 42001, and the internal governance work (inventories, impact assessments, documentation, oversight) that turns rules into practice.',
    why: `AI systems now decide who gets interviewed, which loans are approved and what millions of people read, so governments regulate them and customers ask how they are controlled. Engineers increasingly need to know which rules apply to a system, what documentation and testing they require, and how to build systems that can be audited. This lesson is an overview for learners, not legal advice: the rules differ between countries and change often.`,
    objectives: [
      'Explain the risk-based structure of the EU AI Act and place example systems in its tiers',
      'Describe the obligations for general-purpose AI models and estimate training compute against the 10²⁵ FLOP threshold',
      'Compare governance approaches in the US, the UK, China and international standards',
      'Describe the internal practices that make an AI system governable: inventory, impact assessment, documentation, oversight and monitoring'
    ],
    sections: [
      { type: 'text', title: "What AI governance covers", md: `AI governance is the set of laws, standards and internal practices that decide how AI systems may be built and used, who is responsible when they cause harm, and how anyone can check. The main roles recur everywhere:

- **providers** develop a system or a model and put it on the market;
- **deployers** use it in their own products or processes;
- **affected people** are the job applicants, borrowers, students or patients the system makes decisions about.

Most frameworks share principles first written down in the OECD AI Principles (2019, updated in 2024): respect for human rights and fairness, transparency and explainability, robustness and safety, and accountability. The details differ a great deal between jurisdictions. This lesson describes the situation as of mid-2026 for learners; it is not legal advice, and the rules change often, so check the official texts before relying on any date or threshold.` },
      { type: 'text', title: "The EU AI Act", md: `The EU AI Act (Regulation (EU) 2024/1689) entered into force on 1 August 2024 and applies in stages. It sorts AI uses by risk:

- **Prohibited practices**, banned since February 2025: social scoring by public authorities, manipulation or exploitation of people's vulnerabilities that causes significant harm, untargeted scraping of facial images to build face-recognition databases, emotion recognition in workplaces and schools (with narrow exceptions), predicting crimes from profiling alone, and real-time remote biometric identification in public spaces for law enforcement outside narrowly defined cases.
- **High-risk systems**: AI used for employment, education, credit and essential services, law enforcement, migration and border control, justice, critical infrastructure and biometrics, and AI that is a safety component of regulated products such as medical devices. Providers need risk management, data governance, technical documentation, logging, human oversight, accuracy and robustness testing, a conformity assessment and registration in an EU database; deployers must use the systems as instructed, assign human oversight and monitor them.
- **Transparency duties**: chatbots must make clear that people are talking to an AI, and generated or manipulated images, audio and video ("deepfakes") must be labelled.
- **Minimal risk**: everything else, such as spam filters and video games, with voluntary codes of conduct.

**General-purpose AI models** have their own obligations, applicable since August 2025: technical documentation, information for the companies that build on them, a copyright policy and a public summary of the training content. Models trained with more than 10²⁵ floating-point operations are presumed to pose **systemic risk** and must also be evaluated against such risks, have serious incidents reported and be protected by adequate cybersecurity. Organizations that use AI must also ensure that their staff have sufficient AI literacy.

Most high-risk obligations were scheduled to apply from August 2026, and later for AI inside regulated products; in late 2025 the European Commission proposed postponing parts of them. Fines can reach €35 million or 7% of worldwide annual turnover for prohibited practices.` },
      { type: 'code', title: "Placing systems in the risk tiers", md: `A deliberately simplified classifier that follows the Act's structure (the real rules have many more details and exceptions):`, code: `PROHIBITED = {"social scoring by public authorities", "manipulation that causes harm", "emotion recognition at work or school",
              "untargeted scraping of facial images", "predicting crime from personality traits alone"}
HIGH_RISK_AREAS = {"employment", "education", "credit", "essential public services", "law enforcement",
                   "migration and border control", "critical infrastructure", "justice"}

def tier(system):
    """A simplified, educational reading of the EU AI Act's risk tiers (not legal advice)."""
    if system["purpose"] in PROHIBITED:
        return "prohibited"
    if system["area"] in HIGH_RISK_AREAS and system["decides_about_people"]:
        return "high-risk"
    if system["talks_to_people"] or system["generates_media"]:
        return "transparency obligations"
    return "minimal risk"

systems = [
    {"name": "CV screening tool that ranks applicants", "purpose": "ranking", "area": "employment", "decides_about_people": True, "talks_to_people": False, "generates_media": False},
    {"name": "Customer-service chatbot", "purpose": "support", "area": "retail", "decides_about_people": False, "talks_to_people": True, "generates_media": False},
    {"name": "Spam filter", "purpose": "filtering", "area": "email", "decides_about_people": False, "talks_to_people": False, "generates_media": False},
    {"name": "Credit-scoring model for loans", "purpose": "scoring", "area": "credit", "decides_about_people": True, "talks_to_people": False, "generates_media": False},
    {"name": "Image generator for marketing", "purpose": "creative", "area": "marketing", "decides_about_people": False, "talks_to_people": False, "generates_media": True},
    {"name": "Emotion detection on employees' webcams", "purpose": "emotion recognition at work or school", "area": "employment", "decides_about_people": True, "talks_to_people": False, "generates_media": False},
]
for s in systems:
    print(f"{s['name']:42s} -> {tier(s)}")
print("CV screening is high-risk:", tier(systems[0]) == "high-risk")`, after: `The same technology lands in different tiers depending on its use: a model that ranks people is high-risk in hiring and lending but would be minimal risk for sorting product photos. That is why governance starts with an inventory of *uses*, not of models. The order of the checks matters too: emotion detection at work is banned outright, even though employment is also a high-risk area.`, predict: {"type":"mcq","q":"A company deploys a tool that ranks job applicants' CVs. Which tier does it fall into under this simplified reading of the EU AI Act?","options":["Prohibited","High-risk","Transparency obligations only","Minimal risk"],"answer":1,"explain":"Employment is a high-risk area, and ranking applicants decides about people, so the strict high-risk obligations apply.","expect":["CV screening is high-risk: True"]} },
      { type: 'code', title: "Training compute and the systemic-risk threshold", code: `THRESHOLD = 1e25                                     # the EU AI Act presumes "systemic risk" above this much training compute
print("model                          training compute    vs the 10^25 FLOP threshold")
for name, params, tokens in [("1B parameters, 1T tokens", 1e9, 1e12), ("8B parameters, 15T tokens", 8e9, 15e12),
                             ("70B parameters, 15T tokens", 70e9, 15e12), ("405B parameters, 15.6T tokens", 405e9, 15.6e12),
                             ("1T parameters, 30T tokens", 1e12, 30e12)]:
    flops = 6 * params * tokens                      # about 6 FLOPs per parameter per training token
    print(f"{name:30s} {flops:10.1e} FLOPs     {'above' if flops > THRESHOLD else 'below'}")
print("a 70B model on 15T tokens stays below the threshold:", 6 * 70e9 * 15e12 < THRESHOLD)`, after: `The rule of thumb of 6 FLOPs per parameter per token (2 for the forward pass, 4 for the backward pass) puts a 70-billion-parameter model trained on 15 trillion tokens at about 6 × 10²⁴ FLOPs, below the threshold, and a 405-billion-parameter model on 15.6 trillion tokens at about 4 × 10²⁵, above it. Compute is easy to measure before a model is released, which is why it is used, but it is only a rough proxy for capability: better data and algorithms make the same compute more capable every year, so the Commission can adjust the threshold.`, predict: {"type":"mcq","q":"Does a 70-billion-parameter model trained on 15 trillion tokens exceed the EU AI Act's 10²⁵ FLOP threshold?","options":["Yes, by far","Yes, just barely","No: it needs about 6 × 10²⁴ FLOPs","No: it needs about 10²⁰ FLOPs"],"answer":2,"explain":"6 × 70 × 10⁹ × 15 × 10¹² ≈ 6.3 × 10²⁴ FLOPs, below 10²⁵.","expect":["a 70B model on 15T tokens stays below the threshold: True"]} },
      { type: 'chart', title: "Training compute against model size", md: `Compute from the 6 × parameters × tokens rule for two ways of choosing the amount of training data:`, after: `With a fixed budget of 15 trillion tokens, which recent open models have used, the line is crossed at about 110 billion parameters. At the compute-optimal 20 tokens per parameter, a model needs about 290 billion parameters and 6 trillion tokens to reach it. As training sets keep growing, smaller models cross the line, which is one reason the Act lets the European Commission update the threshold.`, chart: {"kind":"line","x":[100000000,140700000,197900000,278500000,391800000,551300000,775600000,1091000000,1535000000,2160000000,3039000000,4276000000,6016000000,8464000000,11910000000,16750000000,23570000000,33170000000,46660000000,65650000000,92370000000,130000000000,182800000000,257200000000,361900000000,509200000000,716400000000,1008000000000,1418000000000,1995000000000],"xLog":true,"yLog":true,"points":false,"series":[{"name":"20 tokens per parameter (compute-optimal)","y":[1200000000000000000,2376000000000000000,4700000000000000000,9307000000000000000,18420000000000000000,36470000000000000000,72190000000000000000,142800000000000000000,282700000000000000000,559900000000000000000,1.108e+21,2.194e+21,4.343e+21,8.597e+21,1.702e+22,3.367e+22,6.667e+22,1.32e+23,2.613e+23,5.172e+23,1.024e+24,2.028e+24,4.01e+24,7.938e+24,1.572e+25,3.111e+25,6.159e+25,1.219e+26,2.413e+26,4.776e+26]},{"name":"15 trillion tokens","y":[9e+21,1.266e+22,1.781e+22,2.506e+22,3.526e+22,4.962e+22,6.98e+22,9.819e+22,1.382e+23,1.944e+23,2.735e+23,3.848e+23,5.414e+23,7.618e+23,1.072e+24,1.507e+24,2.121e+24,2.985e+24,4.199e+24,5.909e+24,8.313e+24,1.17e+25,1.645e+25,2.315e+25,3.257e+25,4.583e+25,6.448e+25,9.072e+25,1.276e+26,1.795e+26]},{"name":"the five models in the code","x":[1000000000,8000000000,70000000000,405000000000,1000000000000],"y":[6e+21,7.2e+23,6.3e+24,3.791e+25,1.8e+26],"scatter":true}],"xLabel":"parameters (log scale)","yLabel":"training FLOPs (log scale)","hlines":[{"y":1e+25,"label":"EU AI Act threshold, 10²⁵ FLOPs"}]} },
      { type: 'text', title: "The United States, the United Kingdom, China and international standards", md: `- **United States.** There is no comprehensive federal AI law, and federal policy has shifted with each administration: an executive order on AI safety from 2023 was revoked in January 2025, and the 2025 AI Action Plan emphasizes innovation and infrastructure. The **NIST AI Risk Management Framework** (2023), organized around four functions (Govern, Map, Measure, Manage), and its profile for generative AI (2024) are voluntary but widely used. Existing laws on consumer protection and discrimination in hiring and lending still apply to AI. States add their own rules: New York City requires bias audits of automated hiring tools, Colorado passed a broad law on high-risk AI decisions (its start was pushed back into 2026), and California requires developers of the largest frontier models to publish their safety frameworks.
- **United Kingdom.** Existing regulators apply common principles in their own sectors instead of a single AI law, and the AI Security Institute tests frontier models before and after release.
- **China.** Targeted regulations cover recommendation algorithms (2022), "deep synthesis" such as deepfakes (2023) and generative AI services (2023, including registration with regulators), and labelling of AI-generated content has been mandatory since September 2025.
- **International.** The Council of Europe's Framework Convention on AI (2024) is the first binding international treaty on AI; the G7's Hiroshima process produced a code of conduct for developers of advanced AI; and **ISO/IEC 42001** (2023) is a certifiable standard for AI management systems, similar in spirit to ISO 27001 for information security.` },
      { type: 'text', title: "Governance inside an organization", md: `Rules only matter if organizations can follow them, and most of that work is ordinary engineering and management:

- **An inventory** of AI systems: purpose, owner, data, models (including third-party APIs) and who is affected.
- **Risk classification** of each use against the rules that apply.
- **Impact assessments** before deployment and after major changes: who could be harmed and how, which mitigations exist, and what risk remains.
- **Documentation**: datasheets for datasets, model and system cards, evaluation records (see [Responsible AI](#/concept/responsible-ai)).
- **Testing and red teaming**, including for bias and security (see [Evaluating LLM Applications](#/concept/llm-evaluation) and [Adversarial ML, Security & Privacy](#/concept/ml-security)).
- **Human oversight by design**: who can override the system, when, and with what information, while guarding against automation bias (people deferring to the machine).
- **Logging, monitoring and incident response** (see [LLMs in Production](#/concept/llm-observability)).
- **Supplier management**: contracts and assurances from AI vendors.
- **Training**, so that the people using AI understand what it can and cannot do.

Developers of frontier models also publish safety frameworks that define dangerous capability levels and the safeguards they commit to before reaching them.` },
      { type: 'check', question: {"type":"mcq","q":"A company buys a third-party AI tool that scores job applicants and uses it in the EU. Which statement fits the EU AI Act?","options":["It is minimal risk, because the company bought it rather than built it","It is a high-risk use, and the company as deployer has its own obligations, such as human oversight and monitoring","Only the vendor has obligations","It is prohibited"],"answer":1,"explain":"Employment decisions are a high-risk area. Providers carry most obligations, but deployers must use the system as instructed, assign human oversight and monitor it."} }
    ],
    when: `- Before building or buying an AI system that affects people's access to jobs, credit, education or public services.
- When releasing a general-purpose model, or building on one.
- When customers, auditors or regulators ask how your AI is controlled.`,
    pitfalls: `- Treating governance as paperwork at the end: risk classification and documentation shape design decisions.
- Assuming one jurisdiction's rules apply everywhere: obligations depend on where a system is offered and used.
- Relying on dates and thresholds without checking the current official texts: they are revised.
- Confusing this overview with legal advice: involve qualified counsel for real decisions.`,
    takeaways: [
      'The EU AI Act sorts AI uses into prohibited practices, high-risk systems with strict obligations, transparency duties, and minimal risk.',
      'General-purpose AI models carry their own obligations, and models trained with more than 10²⁵ FLOPs are presumed to pose systemic risk.',
      'The US relies on voluntary frameworks, sector regulators and state laws; the UK on existing regulators; China on targeted rules with registration.',
      'Standards such as the NIST AI RMF and ISO/IEC 42001 describe how to manage AI risk as an ongoing process.',
      'Good governance is engineering work: an inventory, impact assessments, documentation, human oversight, logging and monitoring.'
    ],
    quiz: [
      {"type":"mcq","q":"Which practice does the EU AI Act prohibit?","options":["Spam filtering","Social scoring by public authorities","Customer-service chatbots","Product recommendations"],"answer":1,"explain":"Social scoring by public authorities is among the practices banned outright since February 2025."},
      {"type":"mcq","q":"What do the EU AI Act's transparency duties require of a chatbot?","options":["Nothing","Making clear to people that they are interacting with an AI","Publishing its source code","Being certified by a notified body"],"answer":1,"explain":"People must be told they are talking to an AI, and generated media must be labelled."},
      {"type":"fill","q":"With the 6·N·D rule, how many FLOPs does training a 10-billion-parameter model on 1 trillion tokens take? (scientific notation, such as 1e21)","answer":["6e22","6e+22","6.0e22","6 × 10^22","6x10^22"],"explain":"6 × 10¹⁰ × 10¹² = 6 × 10²², well below the 10²⁵ threshold."},
      {"type":"tf","q":"The NIST AI Risk Management Framework is a binding federal law in the United States.","answer":false,"explain":"It is a voluntary framework, though widely used and often referenced in contracts and state rules."},
      {"type":"mcq","q":"What is ISO/IEC 42001?","options":["An EU regulation","A certifiable standard for AI management systems","A benchmark for language models","A file format for model cards"],"answer":1,"explain":"Like ISO 27001 for information security, it describes processes an organization can be audited and certified against."},
      {"type":"mcq","q":"Why do regulators use training compute as a threshold for the most demanding obligations?","options":["Compute perfectly measures danger","It is easy to measure before release, although it is only a rough proxy for capability","Larger models are always unsafe","Compute determines the model's licence"],"answer":1,"explain":"Compute can be counted in advance, which suits regulation, but better data and algorithms keep changing what a given amount of compute can do."}
    ],
    problems: [
      {
        id: 'systemic-risk-compute', title: 'Training Compute and the 10²⁵ FLOP Threshold', difficulty: 'easy',
        prompt: `Training a dense transformer takes about 6 floating-point operations per parameter per training token. Implement \`training_compute(params, tokens, threshold=1e25)\` that returns a tuple \`(flops, above)\`: the estimated training compute 6 · params · tokens, and whether it exceeds the threshold.

\`\`\`text
training_compute(70e9, 15e12)   ->  (6.3e+24, False)
\`\`\``,
        starter: `
def training_compute(params, tokens, threshold=1e25):
    # TODO
    pass
`,
        solution: `
def training_compute(params, tokens, threshold=1e25):
    flops = 6 * params * tokens
    return flops, flops > threshold
`,
        hints: ['Multiply 6 by the number of parameters and the number of tokens.'],
        explain: `The 6·N·D rule counts about 2 FLOPs per parameter per token for the forward pass and 4 for the backward pass. Thresholds based on compute are easy to measure but imperfect: algorithmic progress makes the same compute more capable every year, which is why such thresholds are meant to be revised.`,
        tests: [
          { call: '[round(training_compute(70e9, 15e12)[0] / 1e24, 6), training_compute(70e9, 15e12)[1]]', expect: '[6.3, False]' },
          { name: 'a frontier-scale run', call: 'training_compute(405e9, 15.6e12)[1]', expect: 'True' },
          { name: 'a custom threshold', call: 'training_compute(1e9, 1e12, threshold=1e21)', expect: '(6e+21, True)' },
          { name: 'another model', call: 'training_compute(8e9, 15e12)', hidden: true }
        ]
      },
      {
        id: 'risk-tier', title: 'A Simplified Risk Classifier', difficulty: 'easy',
        prompt: `Implement \`risk_tier(system)\` for a **simplified, educational** reading of the EU AI Act's tiers. \`system\` is a dict with the keys \`"purpose"\`, \`"area"\`, \`"decides_about_people"\`, \`"talks_to_people"\` and \`"generates_media"\`. Use the two sets in the starter code and return, checking in this order:

1. \`"prohibited"\` if the purpose is in PROHIBITED;
2. \`"high-risk"\` if the area is in HIGH_RISK_AREAS and the system decides about people;
3. \`"transparency"\` if it talks to people or generates media;
4. otherwise \`"minimal"\`.`,
        starter: `
PROHIBITED = {"social scoring by public authorities", "manipulation that causes harm", "emotion recognition at work or school"}
HIGH_RISK_AREAS = {"employment", "education", "credit", "essential public services", "law enforcement", "migration and border control", "critical infrastructure", "justice"}

def risk_tier(system):
    # TODO
    pass
`,
        solution: `
PROHIBITED = {"social scoring by public authorities", "manipulation that causes harm", "emotion recognition at work or school"}
HIGH_RISK_AREAS = {"employment", "education", "credit", "essential public services", "law enforcement", "migration and border control", "critical infrastructure", "justice"}

def risk_tier(system):
    if system["purpose"] in PROHIBITED:
        return "prohibited"
    if system["area"] in HIGH_RISK_AREAS and system["decides_about_people"]:
        return "high-risk"
    if system["talks_to_people"] or system["generates_media"]:
        return "transparency"
    return "minimal"
`,
        hints: ['Check the conditions in the given order and return as soon as one applies.'],
        explain: `The order matters: a prohibited purpose stays prohibited even in a high-risk area, and a high-risk chatbot is high-risk, not merely subject to transparency rules. Real classification involves many more details and exceptions, and needs legal review.`,
        tests: [
          { name: 'CV screening', call: 'risk_tier({"purpose": "ranking", "area": "employment", "decides_about_people": True, "talks_to_people": False, "generates_media": False})', expect: "'high-risk'" },
          { name: 'a support chatbot', call: 'risk_tier({"purpose": "support", "area": "retail", "decides_about_people": False, "talks_to_people": True, "generates_media": False})', expect: "'transparency'" },
          { name: 'emotion recognition at work', call: 'risk_tier({"purpose": "emotion recognition at work or school", "area": "employment", "decides_about_people": True, "talks_to_people": False, "generates_media": False})', expect: "'prohibited'" },
          { name: 'a spam filter', call: 'risk_tier({"purpose": "filtering", "area": "email", "decides_about_people": False, "talks_to_people": False, "generates_media": False})', expect: "'minimal'" },
          { name: 'a credit chatbot that decides', call: 'risk_tier({"purpose": "lending", "area": "credit", "decides_about_people": True, "talks_to_people": True, "generates_media": False})', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Regulation (EU) 2024/1689: the Artificial Intelligence Act (official text, EUR-Lex)', url: 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj' },
      { title: 'NIST, AI Risk Management Framework', url: 'https://www.nist.gov/itl/ai-risk-management-framework' },
      { title: 'OECD, AI Principles', url: 'https://oecd.ai/en/ai-principles' }
    ]
  },

  // =====================================================================================
  {
    id: 'contextual-bandits', branch: 'spec', title: 'Contextual Bandits & Online Experimentation', short: 'Contextual Bandits', level: 3, minutes: 45, prereqs: ['reinforcement-learning', 'stat-inference'],
    tags: ['bandits', 'multi-armed bandit', 'contextual bandit', 'regret', 'exploration', 'exploitation', 'epsilon-greedy', 'ucb', 'thompson sampling', 'linucb', 'off-policy evaluation', 'inverse propensity scoring', 'experimentation', 'a/b testing'],
    summary: 'Learning from interaction one decision at a time: regret, epsilon-greedy, upper confidence bounds and Thompson sampling, contextual bandits that choose per user, evaluating a new policy from logged data with inverse propensity scoring, and the practical side of online experimentation (delayed rewards, changing behaviour, guardrail metrics).',
    why: `Which headline, recommendation, price or layout should the next visitor see? A fixed A/B test explores evenly for weeks and then exploits; a bandit shifts traffic towards what works while it is still learning, and a contextual bandit learns what works for whom. Bandits run much of the web's personalization, and their logged data raise a subtle question: how do you evaluate a new policy without deploying it?`,
    objectives: [
      'Define regret, and compare epsilon-greedy, UCB and Thompson sampling',
      'Explain how a contextual bandit uses features of the situation to choose actions',
      'Estimate a new policy\'s reward from logged data with inverse propensity scoring',
      'Recognize the practical pitfalls of online experimentation'
    ],
    sections: [
      { type: 'text', title: "Between A/B tests and reinforcement learning", md: `An A/B test splits traffic evenly between options for a fixed period, then picks a winner. While it runs, half the visitors see the worse option, which is the price of a clean measurement. A **multi-armed bandit** learns while it serves: every visitor is a decision, every click is feedback, and traffic shifts towards the better options as evidence accumulates. A **contextual bandit** also looks at features of the situation (device, time of day, past behaviour) and learns which option is best for whom.

Bandits sit between supervised learning and full reinforcement learning (see [Reinforcement Learning](#/concept/reinforcement-learning)): actions have consequences you only observe for the action you took, but they do not change future states. The quantity to keep small is the **regret**:

$$ Regret(T) = Σ_{t=1..T} ( μ* − μ_{a_t} ) $$
:::where
T — the number of decisions so far
a_t — the action chosen at step t
μ_{a_t} — the expected reward of that action, for example its click-through rate
μ* — the expected reward of the best action
Regret(T) — the reward lost, in expectation, by not always choosing the best action
:::` },
      { type: 'text', title: "Three ways to balance exploring and exploiting", md: `- **Epsilon-greedy**: show the option with the best observed average, except with probability ε, when you pick an option at random. Simple, but it keeps exploring at the same rate forever, so its regret grows in a straight line.
- **Upper confidence bound (UCB)**: be optimistic about options you know little about. Pick the option with the highest observed mean plus an uncertainty bonus that shrinks as the option is tried:

$$ a_t = argmax_a ( μ̂_a + √( 2 ln t / n_a ) ) $$
:::where
μ̂_a — the observed average reward of option a
n_a — how often option a has been chosen so far
t — the current step
√(2 ln t / n_a) — the exploration bonus: large for rarely tried options, slowly growing with time
:::

- **Thompson sampling**: keep a posterior distribution for each option's reward rate, here a Beta distribution updated with clicks and non-clicks (see [Bayesian Inference](#/concept/bayesian-methods)); draw one plausible rate from each posterior and pick the option with the highest draw. Each option is chosen with exactly the probability that it is the best, given the evidence so far.` },
      { type: 'code', title: "Epsilon-greedy, UCB and Thompson sampling", md: `Five headlines with click-through rates between 2% and 10% (unknown to the learner), 5,000 visitors, and each strategy run 200 times:`, code: `import numpy as np
rng = np.random.default_rng(0)
p = np.array([0.02, 0.04, 0.10, 0.05, 0.07])         # true click-through rates of five headlines (unknown to the agent)
K, T, runs = len(p), 5000, 200

def play(strategy):
    n, s = np.zeros((runs, K)), np.zeros((runs, K))  # visits and clicks per headline, for 200 independent runs at once
    regret, rows = np.zeros(T), np.arange(runs)
    for t in range(T):
        if strategy == "epsilon-greedy (10%)":
            a = np.where(rng.random(runs) < 0.1, rng.integers(K, size=runs), np.argmax(s / np.maximum(n, 1), axis=1))
        elif strategy == "UCB":
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(n, 1))
            a = np.argmax(np.where(n == 0, np.inf, s / np.maximum(n, 1) + bonus), axis=1)
        else:                                        # Thompson sampling: draw a plausible rate per headline, show the best draw
            a = np.argmax(rng.beta(1 + s, 1 + n - s), axis=1)
        click = rng.random(runs) < p[a]
        n[rows, a] += 1
        s[rows, a] += click
        regret[t] = (p.max() - p[a]).mean()           # clicks lost, on average, by not showing the best headline
    return np.cumsum(regret)

result = {name: play(name) for name in ("epsilon-greedy (10%)", "UCB", "Thompson sampling")}
for name, r in result.items():
    print(f"{name:22s} clicks lost after 1,000 visitors {r[999]:5.1f}, after 5,000 {r[-1]:5.1f}")
print("lowest regret:", min(result, key=lambda k: result[k][-1]))`, after: `Thompson sampling loses the fewest clicks, and its advantage grows with time. Epsilon-greedy keeps showing a random headline to one visitor in ten forever, so its losses grow steadily. UCB does worst here: with click rates this low, the exploration bonus is large compared with the differences between headlines, so it keeps trying poor ones for a long time (its losses grow more slowly later on). Which strategy wins depends on the problem, but Thompson sampling is a strong, simple default.`, predict: {"type":"mcq","q":"Which strategy will have lost the fewest clicks after 5,000 visitors?","options":["Epsilon-greedy, because it exploits 90% of the time","UCB, because it is optimistic","Thompson sampling","All three lose the same"],"answer":2,"explain":"Thompson sampling explores only as much as its uncertainty justifies, and stops wasting visitors on headlines that are clearly worse. Epsilon-greedy keeps exploring at random forever, and UCB's bonus stays large for a long time when click rates are this low.","expect":["lowest regret: Thompson sampling"]} },
      { type: 'chart', title: "Clicks lost as visitors arrive", md: `The cumulative regret from the code above, averaged over its 200 runs:`, after: `Epsilon-greedy's losses settle into a straight line, because one visitor in ten always sees a random headline. Thompson sampling's curve keeps bending over as its posteriors narrow. UCB's rises steeply for thousands of visitors, because with click rates this low its exploration bonus stays large compared with the gaps between headlines; in theory its losses eventually grow only with the logarithm of the number of visitors, but that takes far longer than 5,000 visitors here.`, chart: {"kind":"line","x":[1,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000],"points":false,"series":[{"name":"epsilon-greedy (10%)","y":[0.0772,6.652,11.59,15.62,18.84,21.44,23.72,25.82,27.64,29.32,30.88,32.38,33.77,35.08,36.31,37.48,38.6,39.71,40.84,41.92,42.96,44,44.99,45.93,46.86,47.76,48.64,49.49,50.29,51.06,51.85,52.62,53.36,54.11,54.84,55.54,56.21,56.89,57.6,58.27,58.9,59.55,60.2,60.85,61.49,62.13,62.76,63.41,64.04,64.65,65.23]},{"name":"UCB","y":[0.08,4.194,8.244,12.21,16.07,19.92,23.73,27.42,31.08,34.69,38.27,41.86,45.29,48.79,52.3,55.73,59.03,62.32,65.65,68.99,72.31,75.54,78.83,82.03,85.21,88.36,91.44,94.6,97.65,100.7,103.6,106.5,109.5,112.5,115.5,118.4,121.3,124.2,126.9,129.6,132.4,135.3,138.1,140.9,143.5,146.3,149.1,151.8,154.5,157.1,159.7]},{"name":"Thompson sampling","y":[0.0445,3.838,6.945,9.606,11.87,13.86,15.67,17.26,18.68,19.96,21.1,22.15,23.08,23.97,24.78,25.56,26.29,27.01,27.7,28.33,28.9,29.47,30.01,30.55,31.05,31.51,31.96,32.4,32.8,33.19,33.58,33.94,34.26,34.59,34.91,35.23,35.53,35.81,36.08,36.35,36.6,36.84,37.07,37.31,37.53,37.75,37.96,38.17,38.39,38.62,38.82]}],"xLabel":"visitors so far","yLabel":"clicks lost (cumulative regret)","yMin":0} },
      { type: 'text', title: "Adding context", md: `A plain bandit looks for the single best option for everyone. When visitors differ, the best option depends on the **context**: phone or desktop, new or returning, morning or night. Contextual bandits learn a model of reward given context and action:

- **Separate bandits per segment**: simple when there are a few large, known groups.
- **LinUCB**: a linear model of reward for each action (ridge regression on the context features), plus a confidence bonus from the uncertainty of its predictions, which shares information between similar contexts.
- **Neural and tree-based bandits**: any model that gives a prediction and an uncertainty estimate, for example with Thompson sampling over an ensemble or a Bayesian last layer.

The more the best action varies with context, the more a contextual bandit gains over a plain one.` },
      { type: 'code', title: "One layout for everyone, or one per group?", md: `Phone users prefer layout 0 and desktop users layout 2. A plain Thompson-sampling bandit has to pick one layout for everyone; a contextual one keeps separate statistics per device:`, code: `import numpy as np
rng = np.random.default_rng(1)
# Two kinds of visitors prefer different page layouts: phone users like layout 0, desktop users layout 2
p = {"phone": np.array([0.09, 0.05, 0.04]), "desktop": np.array([0.04, 0.05, 0.08])}
T, runs = 4000, 200
def play(use_context):
    clicks = 0.0
    n, s = {}, {}
    for key in (["phone", "desktop"] if use_context else ["all"]):
        n[key], s[key] = np.zeros((runs, 3)), np.zeros((runs, 3))
    rows = np.arange(runs)
    for t in range(T):
        group = "phone" if rng.random() < 0.6 else "desktop"
        key = group if use_context else "all"
        a = np.argmax(rng.beta(1 + s[key], 1 + n[key] - s[key]), axis=1)   # Thompson sampling within the key
        click = rng.random(runs) < p[group][a]
        n[key][rows, a] += 1
        s[key][rows, a] += click
        clicks += click.mean()
    return clicks
plain, contextual = play(False), play(True)
best = T * (0.6 * 0.09 + 0.4 * 0.08)
print(f"clicks from {T:,} visitors: one layout for everyone {plain:.0f}, a layout per group {contextual:.0f}, best possible {best:.0f}")
print("using the context earns more clicks:", contextual > plain * 1.1)`, after: `The plain bandit settles on layout 0, the best on average, and gives desktop users a layout they like least. Keeping separate statistics per device earns about 20% more clicks and comes much closer to the best possible. Its cost is slower learning: each group's statistics see only part of the traffic, which is why richer contexts need models that share information, such as LinUCB.`, predict: {"type":"mcq","q":"Phone users prefer layout 0 and desktop users layout 2. Compared with one bandit for everyone, how will a bandit that keeps separate statistics per device do?","options":["Worse, because it learns from less data per group","About the same","Clearly better, because it can show each group its preferred layout","It cannot learn at all"],"answer":2,"explain":"With different best actions per group, the context is worth far more than the data it costs: about 312 clicks against 259 here.","expect":["using the context earns more clicks: True"]} },
      { type: 'text', title: "Evaluating a new policy from logged data", md: `Suppose the bandit has been running for months and you want to know whether a new policy would do better, without exposing users to it. The logs contain, for each decision, the context, the action shown, the reward, and (if you were careful) the **propensity**: the probability with which the logging policy chose that action. **Inverse propensity scoring** reweights the logged rewards:

$$ V̂(π) = (1 / n) Σ_i r_i · 1[π(x_i) = a_i] / p_i $$
:::where
π — the new policy, which maps a context to an action
x_i, a_i, r_i — the context, the action shown and the reward of logged decision i
p_i — the propensity: the probability that the logging policy chose a_i in context x_i
1[π(x_i) = a_i] — 1 when the new policy would have chosen the same action, otherwise 0
n — the number of logged decisions
V̂(π) — the estimated average reward of the new policy
:::

The estimate is unbiased, provided the logging policy gave every action the new policy might choose a non-zero probability. Its weakness is variance: rarely logged actions get very large weights. In practice the weights are capped, or the estimate is combined with a model of the reward (**doubly robust** estimation). The lesson for engineering is simple: always log the propensities, and never let the production policy become fully deterministic if you want to evaluate alternatives later.` },
      { type: 'text', title: "Running bandits in production", md: `- **Delayed rewards.** Purchases or subscriptions arrive hours or days after the decision, so updates must match rewards to the decisions that caused them.
- **Changing behaviour.** Preferences drift and novelty wears off: weight recent data more (discounting, sliding windows) and keep some exploration.
- **Short-term proxies.** Optimizing clicks can hurt satisfaction or retention. Track guardrail metrics alongside the reward, and check long-term effects with occasional classic A/B tests.
- **Batching.** Many systems update their statistics every few minutes rather than after every visitor.
- **Fairness and ethics.** Bandits that set prices or allocate opportunities can treat groups differently; review what the policy learns, not only what it earns.
- **Statistics.** Adaptively collected data violate the assumptions of a standard significance test, so use methods designed for sequential or adaptive experiments (see [Statistical Inference](#/concept/stat-inference)).` },
      { type: 'check', question: {"type":"mcq","q":"A team wants to evaluate new recommendation policies offline from its logs. What must it record for every decision?","options":["Only the clicks","The context, the action shown, the reward and the probability with which that action was chosen","Only the user's id","The model's weights"],"answer":1,"explain":"Without the propensities, the logged rewards cannot be reweighted correctly, and offline estimates are biased towards whatever the old policy preferred."} }
    ],
    when: `- Many repeated decisions with quick feedback: headlines, recommendations, notifications, ad creatives.
- The cost of showing a worse option during a long test is high.
- Prefer a classic A/B test when you need a clean, unbiased estimate of a long-term effect for one big decision.`,
    pitfalls: `- Not logging the probability with which each action was chosen: without it, offline evaluation is biased.
- Optimizing a short-term proxy (clicks) that hurts the long-term goal (satisfaction, retention).
- Assuming the world stands still: user behaviour drifts, so old evidence should count less.
- Stopping exploration entirely, so that a better option is never discovered.`,
    takeaways: [
      'Regret is the reward lost compared with always choosing the best action; good strategies keep it growing slowly.',
      'UCB adds an optimism bonus for uncertain actions; Thompson sampling plays each action with the probability that it is the best.',
      'Contextual bandits learn which action is best in each situation, which helps whenever preferences differ between users.',
      'Inverse propensity scoring evaluates a new policy from logs by reweighting the rewards of matching actions.',
      'Log propensities, watch guardrail metrics, and account for delays and drift.'
    ],
    quiz: [
      {"type":"mcq","q":"What does a bandit's regret measure?","options":["How often the model crashed","The reward lost by not always choosing the best action","The number of actions","The time spent training"],"answer":1,"explain":"Regret sums, over all decisions, the gap between the best action's expected reward and that of the action chosen."},
      {"type":"tf","q":"With a fixed ε, epsilon-greedy's regret eventually grows in a straight line, because it keeps exploring at the same rate forever.","answer":true,"explain":"A fixed share of decisions stays random, each costing on average the gap to the best action; decaying ε over time avoids this."},
      {"type":"mcq","q":"How does Thompson sampling choose an action?","options":["Always the best observed average","At random","It draws a plausible reward rate for each action from its posterior and picks the highest draw","It picks the least tried action"],"answer":2,"explain":"Actions are chosen with the probability that they are the best, so uncertain actions still get explored, in proportion to how promising they are."},
      {"type":"fill","q":"A logged action had propensity 0.25 and reward 1, and the new policy would choose the same action. What does this entry contribute to the sum in the IPS estimate, before averaging?","answer":["4","4.0"],"explain":"r / p = 1 / 0.25 = 4: rarely logged actions are up-weighted to stand in for the decisions the logging policy did not make."},
      {"type":"mcq","q":"When does a contextual bandit gain the most over a plain bandit?","options":["When every user prefers the same option","When the best option differs between kinds of users or situations","When there is only one option","When rewards are never observed"],"answer":1,"explain":"If everyone prefers the same option, context adds nothing; the more preferences differ, the more there is to gain."},
      {"type":"mcq","q":"Why should a production bandit keep some randomness if the team wants to evaluate other policies later?","options":["Randomness always increases reward","Off-policy estimates need every action a new policy might take to have a non-zero logged probability","It makes the logs smaller","Regulators require it"],"answer":1,"explain":"If the logging policy never shows an action, no reweighting can tell you what would have happened."}
    ],
    problems: [
      {
        id: 'ucb-choice', title: 'Choose an Arm with UCB', difficulty: 'easy',
        prompt: `Implement \`ucb_choice(counts, rewards, t, c=2.0)\` for the UCB1 strategy. \`counts[i]\` is how often arm i has been played, \`rewards[i]\` the total reward it earned, and \`t\` the current round (t ≥ 1).

- If any arm has never been played, return the lowest such index.
- Otherwise return the index with the largest mean reward + √(c · ln t / counts[i]) (the first one on a tie).`,
        starter: `
import math

def ucb_choice(counts, rewards, t, c=2.0):
    # TODO
    pass
`,
        solution: `
import math

def ucb_choice(counts, rewards, t, c=2.0):
    for i, n in enumerate(counts):
        if n == 0:
            return i
    scores = [r / n + math.sqrt(c * math.log(t) / n) for n, r in zip(counts, rewards)]
    return scores.index(max(scores))
`,
        hints: ['Play every arm once before using the formula.', 'The bonus shrinks as an arm is played more often.'],
        explain: `The bonus is a confidence bound: an arm that has rarely been tried could still be the best, so it gets the benefit of the doubt. As evidence accumulates, the bonus shrinks and the choice follows the observed means.`,
        tests: [
          { name: 'untried arms first', call: 'ucb_choice([3, 0, 2], [1, 0, 1], 5)', expect: '1' },
          { name: 'the bonus favours a rarely tried arm', call: 'ucb_choice([100, 2], [60, 1], 102)', expect: '1' },
          { name: 'with equal counts, the best mean wins', call: 'ucb_choice([10, 10, 10], [2, 7, 5], 30)', expect: '1' },
          { name: 'random state vs reference', setup: 'import random\nrandom.seed(13)\nns = [random.randint(1, 50) for _ in range(6)]\nrs = [random.uniform(0, n) for n in ns]', call: 'ucb_choice(ns, rs, sum(ns))', hidden: true }
        ]
      },
      {
        id: 'ips-estimate', title: 'Off-Policy Evaluation with IPS', difficulty: 'medium',
        prompt: `A logging policy chose actions with known probabilities. Each log entry is a tuple \`(context, action, reward, propensity)\`, where \`propensity\` is the probability with which the logging policy chose that action in that context. Implement \`ips_estimate(logs, new_policy)\`, where \`new_policy(context)\` returns the action a new deterministic policy would take:

estimate = (1 / n) Σ reward · [new_policy(context) == action] / propensity

\`\`\`text
logs = [("a", 0, 1.0, 0.5), ("a", 1, 0.0, 0.5), ("b", 1, 1.0, 0.5), ("b", 0, 0.0, 0.5)]
ips_estimate(logs, lambda ctx: 0 if ctx == "a" else 1)   ->  1.0
\`\`\``,
        starter: `
def ips_estimate(logs, new_policy):
    # TODO
    pass
`,
        solution: `
def ips_estimate(logs, new_policy):
    total = 0.0
    for context, action, reward, propensity in logs:
        if new_policy(context) == action:
            total += reward / propensity
    return total / len(logs)
`,
        hints: ['Only entries where the new policy agrees with the logged action contribute.', 'Divide each matching reward by its propensity, then average over all entries.'],
        explain: `Dividing by the propensity corrects for how often the logging policy happened to try each action, which makes the estimate unbiased. Its weakness is variance: actions that were logged rarely get huge weights, so in practice the weights are capped or combined with a reward model (doubly robust estimation).`,
        tests: [
          { call: 'ips_estimate([("a", 0, 1.0, 0.5), ("a", 1, 0.0, 0.5), ("b", 1, 1.0, 0.5), ("b", 0, 0.0, 0.5)], lambda ctx: 0 if ctx == "a" else 1)', expect: '1.0' },
          { name: 'a worse policy', call: 'ips_estimate([("a", 0, 1.0, 0.5), ("a", 1, 0.0, 0.5), ("b", 1, 1.0, 0.5), ("b", 0, 0.0, 0.5)], lambda ctx: 1 if ctx == "a" else 0)', expect: '0.0' },
          { name: 'rarely logged actions weigh more', call: 'round(ips_estimate([("x", 1, 1.0, 0.25), ("x", 0, 0.0, 0.75)], lambda ctx: 1), 6)', expect: '2.0' },
          { name: 'random logs vs reference', setup: 'import random\nrandom.seed(21)\nlogs = []\nfor _ in range(200):\n    ctx = random.choice(["m", "d"]); a = random.choice([0, 1, 2])\n    logs.append((ctx, a, float(random.random() < (0.3 if a == (0 if ctx == "m" else 2) else 0.1)), 1 / 3))', call: 'round(ips_estimate(logs, lambda ctx: 0 if ctx == "m" else 2), 6)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Lattimore & Szepesvári, Bandit Algorithms (book, free PDF)', url: 'https://tor-lattimore.com/downloads/book/book.pdf' },
      { title: 'Russo et al., A Tutorial on Thompson Sampling (2017)', url: 'https://arxiv.org/abs/1707.02038' },
      { title: 'Li et al., A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB, 2010)', url: 'https://arxiv.org/abs/1003.0146' }
    ]
  },

  // =====================================================================================
  {
    id: 'geometric-dl', branch: 'spec', title: 'Geometric & Equivariant Deep Learning', short: 'Geometric Deep Learning', level: 3, minutes: 45, prereqs: ['gnn', 'cnn'],
    tags: ['geometric deep learning', 'symmetry', 'invariance', 'equivariance', 'translation equivariance', 'permutation invariance', 'deep sets', 'rotation equivariance', 'e(3)', 'egnn', 'molecules', 'point clouds', 'data augmentation'],
    summary: 'Building known symmetries into neural networks: the difference between invariance and equivariance, why convolutions are translation-equivariant and graph networks permutation-equivariant, Deep Sets for unordered inputs, rotation- and translation-equivariant networks for molecules and 3D data, and when building symmetry in beats learning it from augmented data.',
    why: `Many inputs come with symmetries: a cat is a cat wherever it sits in the image, a molecule's energy does not change when you rotate it, and a set of points has no natural order. A network that has to learn these facts from data wastes capacity and data, and may still get them slightly wrong. Building the symmetry into the architecture makes models more data-efficient and physically consistent, and it explains, in one framework, why CNNs, GNNs and transformers look the way they do.`,
    objectives: [
      'Define invariance and equivariance, and test a function for both',
      'Explain why convolutions are translation-equivariant and message passing is permutation-equivariant',
      'Build rotation-invariant features from distances, and an equivariant coordinate update',
      'Decide between building a symmetry into the model and teaching it with data augmentation'
    ],
    sections: [
      { type: 'text', title: "Symmetry as prior knowledge", md: `Many problems come with transformations that should not matter. Shifting a photo does not change what is in it, rotating a molecule does not change its energy, and reordering the points of a point cloud does not change the shape. Two properties describe how a function should behave under such a transformation g:

$$ invariant: f(g · x) = f(x)          equivariant: f(g · x) = g · f(x) $$
:::where
x — the input, such as an image, a graph or a set of atom positions
g · x — the input after a transformation g: a shift, a rotation, a reordering
f — the network or one of its layers
:::

An image classifier should be **invariant** to shifts (the label stays the same), while a segmentation network should be **equivariant** (shift the input and the mask shifts with it). The idea of *geometric deep learning* is that the major architectures are each built around a symmetry:

- **CNNs**: translations, because the same filter slides across the image;
- **graph neural networks**: permutations of the nodes, because messages are summed over neighbours (see [Graph Neural Networks](#/concept/gnn));
- **transformers without position encodings**: permutations of the tokens, which is why positions have to be added explicitly (see [Positional Encoding](#/concept/positional-encoding));
- **rotation-equivariant networks**: rotations and translations in 3D, for molecules, proteins and point clouds.` },
      { type: 'code', title: "Convolutions commute with shifts", code: `import numpy as np
x = np.zeros(32); x[5:9] = [1, 2, 3, 1]              # a 1-D signal with a small pattern
kernel = np.array([1.0, -2.0, 1.0])
conv = lambda v: np.convolve(v, kernel, mode="same")  # the same small filter slides along the signal
W = np.random.default_rng(0).normal(size=(32, 32))
dense = lambda v: W @ v                               # a fully connected layer: every output has its own weights
shift = lambda v: np.roll(v, 10)                      # move the pattern 10 places to the right
print("convolution: shift then filter == filter then shift:", bool(np.allclose(conv(shift(x)), shift(conv(x)))))
print("dense layer: shift then transform == transform then shift:", bool(np.allclose(dense(shift(x)), shift(dense(x)))))`, after: `The convolution gives the same response wherever the pattern appears, just moved along with it, because it uses the same weights at every position. The dense layer has separate weights for every position, so a shifted pattern produces an unrelated output, and it would need to see the pattern in every position during training to handle it. Weight sharing is equivariance built in, and it is why CNNs learn from far fewer images than fully connected networks.`, predict: {"type":"mcq","q":"The pattern is moved 10 positions to the right. What happens to the convolution's output?","options":["It changes completely","It moves 10 positions to the right and is otherwise identical","It stays exactly where it was","It becomes zero"],"answer":1,"explain":"Convolution is translation-equivariant: filtering a shifted signal gives the shifted filter output. A dense layer has no such property.","expect":["convolution: shift then filter == filter then shift: True","dense layer: shift then transform == transform then shift: False"]} },
      { type: 'text', title: "Sets and graphs: permutation symmetry", md: `When the input is a **set**, such as the points of a point cloud, the items in a shopping basket or the atoms of a molecule, the order in which elements are listed carries no information. **Deep Sets** (2017) showed that essentially every permutation-invariant function can be written as

$$ f(X) = ρ( Σ_{x ∈ X} φ(x) ) $$
:::where
X — the set of elements
φ — a network applied to each element separately
Σ — a sum (or mean, or max) over the set, which ignores order
ρ — a network applied to the pooled result
:::

Graph neural networks follow the same recipe with neighbourhoods: each node combines its neighbours' messages with a sum or a mean, so renumbering the nodes renumbers the outputs and nothing else. Attention without positional information is also permutation-equivariant, which makes transformers natural set and graph models once positions or edges are supplied as features.` },
      { type: 'text', title: "Rotations and translations in 3D", md: `Molecules, proteins and point clouds live in 3D space, where the relevant symmetries are rotations, translations and sometimes reflections (together called E(3)). There are two ways to respect them:

- **Invariant features.** Distances and angles between points do not change when the whole object is rotated or moved, so a network that only sees them is invariant by construction. SchNet and many molecular models work this way.
- **Equivariant updates.** When the output is itself geometric, such as forces on atoms or new coordinates, the network must rotate its outputs along with the input. EGNN (2021) updates each point by moving it along the directions to its neighbours, weighted by functions of their distances; tensor-field networks, NequIP and MACE use spherical harmonics to carry richer directional information.

Equivariant networks trained on quantum-chemistry data now predict interatomic forces accurately enough to run molecular simulations much faster than the calculations they learned from, and AlphaFold 2's structure module used attention that is invariant to rotating and moving the protein (see [AI for Science](#/concept/ai-for-science)).` },
      { type: 'code', title: "Invariant and equivariant by construction", md: `A 20-atom "molecule", rotated and shifted. Three functions: one that reads raw coordinates, one that reads only distances, and one equivariant update:`, code: `import numpy as np
rng = np.random.default_rng(1)
pts = rng.normal(size=(20, 3))                       # 20 atoms of a molecule, with x, y, z coordinates
q, r = np.linalg.qr(rng.normal(size=(3, 3)))
R = q * np.sign(np.diag(r))
if np.linalg.det(R) < 0:
    R[:, 0] = -R[:, 0]                               # a proper rotation (no mirror image)
t = rng.normal(size=3)
moved = pts @ R.T + t                                # the same molecule, rotated and shifted

w = np.random.default_rng(2).normal(size=60) * 0.1
raw_model = lambda p: float(np.tanh(p.reshape(-1) @ w))                     # reads raw coordinates
def distance_model(p):                                                        # reads only distances between atoms
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return float(np.tanh(np.exp(-d).sum() / 100))
def equivariant_step(p):                                                      # moves each atom along differences to others
    diff = p[:, None] - p[None]
    d = np.linalg.norm(diff, axis=-1, keepdims=True)
    return p + 0.1 * (diff * np.exp(-d)).sum(axis=1)

print(f"raw-coordinate model:  {raw_model(pts):+.4f} before, {raw_model(moved):+.4f} after moving the molecule")
print(f"distance-based model:  {distance_model(pts):+.4f} before, {distance_model(moved):+.4f} after")
gap = np.abs(equivariant_step(moved) - (equivariant_step(pts) @ R.T + t)).max()
print(f"equivariant update: moving the molecule before or after the update differs by {gap:.1e}")
print("invariant and equivariant by construction:", abs(distance_model(pts) - distance_model(moved)) < 1e-9 and gap < 1e-9)`, after: `The raw-coordinate model gives a completely different answer for the same molecule in a new pose, so it would have to learn from many rotated examples that orientation does not matter. The distance-based model cannot tell the difference at all. The equivariant update gives the same result whether the molecule is moved before or after it, up to rounding error at the level of 10⁻¹⁶: its outputs turn and shift exactly with its input. Neither property had to be learned.`, predict: {"type":"mcq","q":"The molecule is rotated and shifted in space. Which model's output changes?","options":["Only the distance-based model","Only the raw-coordinate model","Both","Neither"],"answer":1,"explain":"Distances between atoms do not change under rotations and translations, so the distance-based model is invariant by construction; a model reading raw coordinates is not.","expect":["invariant and equivariant by construction: True"]} },
      { type: 'text', title: "Build the symmetry in, or learn it from data?", md: `- **Built-in symmetry** saves data and guarantees consistency: a force field that is exactly equivariant can never predict that a molecule spins up on its own. It matters most with small datasets and physical systems.
- **Data augmentation** (training on randomly rotated, shifted or reordered copies) is simpler and works with any architecture, but the symmetry is then only approximate and has to be learned. Very large models often take this route: AlphaFold 3 replaced AlphaFold 2's symmetry-aware structure module with a diffusion module that is not equivariant and is trained with random rotations instead.
- **Check that the symmetry is real.** Mirror images of a molecule (chirality) can behave very differently, so a model should not be reflection-invariant there; up and down matter for images of scenes; the order of words matters in text.
- **Test it numerically**, as in the code above: transform the input and compare the outputs. Bugs in "equivariant" code are common and silent.` },
      { type: 'check', question: {"type":"mcq","q":"A segmentation network must output a mask for every pixel. With respect to shifting the image, it should be…","options":["Invariant: the mask should stay the same","Equivariant: the mask should shift along with the image","Neither","Both at once"],"answer":1,"explain":"The output is defined per position, so it must move with the input; a single label for the whole image would instead be invariant."} }
    ],
    when: `- Molecules, proteins, materials and point clouds, where rotations and translations should not change predictions.
- Sets and graphs, where the order of elements carries no meaning.
- Small datasets, where a built-in symmetry replaces data you do not have.`,
    pitfalls: `- Assuming a symmetry the problem does not have: chirality, gravity or reading direction can make orientation matter.
- Checking equivariance only on paper: test it numerically, as in the code below.
- Paying a large architectural cost for a symmetry that cheap data augmentation would teach well enough.`,
    takeaways: [
      'Invariant: transforming the input leaves the output unchanged. Equivariant: the output transforms along with the input.',
      'Convolutions share weights across positions, which makes them translation-equivariant; message passing makes GNNs permutation-equivariant.',
      'Distances between points are unchanged by rotations, translations and reordering, so models built on them are invariant by construction.',
      'Equivariant networks update vectors along differences between points, so rotating the input rotates the output.',
      'Built-in symmetry saves data; augmentation is simpler and lets very large models learn approximate symmetry themselves.'
    ],
    quiz: [
      {"type":"mcq","q":"What makes a convolutional layer translation-equivariant?","options":["Its activation function","The same weights are applied at every position","Pooling","Its large number of parameters"],"answer":1,"explain":"Weight sharing means a pattern produces the same response wherever it appears, shifted along with it."},
      {"type":"tf","q":"A network that only uses distances between atoms gives the same output when the whole molecule is rotated.","answer":true,"explain":"Rotations and translations preserve distances, so such a network is invariant by construction."},
      {"type":"mcq","q":"In Deep Sets, f(X) = ρ(Σ φ(x)). Why does this ignore the order of the elements?","options":["Because φ is linear","Because a sum gives the same result in any order","Because ρ sorts its input","Because sets are small"],"answer":1,"explain":"Summing (or averaging, or taking the maximum) over the elements is symmetric, so reordering them cannot change the result."},
      {"type":"mcq","q":"Why should a model of molecular properties often not be invariant to reflections?","options":["Reflections are impossible","Mirror-image molecules (enantiomers) can have very different biological effects","Reflections change the number of atoms","It is too expensive"],"answer":1,"explain":"Chirality matters in chemistry and biology, so reflection invariance would erase real information."},
      {"type":"mcq","q":"What is the main alternative to building a symmetry into the architecture?","options":["Using a bigger learning rate","Data augmentation: training on transformed copies of the data","Removing features","Using fewer layers"],"answer":1,"explain":"Augmentation teaches an approximate symmetry from data and works with any architecture, at the cost of more data and training."},
      {"type":"tf","q":"Attention without any positional information treats its input tokens as an unordered set.","answer":true,"explain":"Without positions, permuting the tokens just permutes the outputs, so the order has to be supplied through positional encodings."}
    ],
    problems: [
      {
        id: 'symmetry-check', title: 'Invariant, Equivariant or Neither?', difficulty: 'medium',
        prompt: `Bugs in "equivariant" code are common and silent, so it pays to test symmetries numerically. Implement \`symmetry_check(f, g, inputs, tol=1e-9)\`, where \`f\` is a function (a layer or a model) and \`g\` a transformation, both taking and returning lists of numbers. Return:

- \`"invariant"\` if f(g(x)) equals f(x) for every x in \`inputs\`;
- otherwise \`"equivariant"\` if f(g(x)) equals g(f(x)) for every x;
- otherwise \`"neither"\`.

Two lists are equal when they have the same length and every pair of elements differs by at most \`tol\`.

\`\`\`text
roll = lambda v: v[-1:] + v[:-1]                        # shift one place to the right, wrapping around
symmetry_check(lambda v: [sum(v)], roll, [[1, 2, 3]])   ->  "invariant"
\`\`\``,
        starter: `
def symmetry_check(f, g, inputs, tol=1e-9):
    # TODO
    pass
`,
        solution: `
def symmetry_check(f, g, inputs, tol=1e-9):
    def same(a, b):
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
    if all(same(f(g(x)), f(x)) for x in inputs):
        return "invariant"
    if all(same(f(g(x)), g(f(x))) for x in inputs):
        return "equivariant"
    return "neither"
`,
        hints: ['Compare f(g(x)) with f(x) for invariance, and with g(f(x)) for equivariance.', 'Write a small helper that compares two lists element by element within tol.'],
        explain: `This is the numerical test from the lesson: transform the input and compare the outputs. Real checks use random inputs and random transformations (rotations, permutations, shifts) and run on every new layer, because one misplaced index can silently break a symmetry that the architecture is supposed to guarantee.`,
        tests: [
          { setup: 'roll = lambda v: v[-1:] + v[:-1]', call: 'symmetry_check(lambda v: [sum(v)], roll, [[1, 2, 3]])', expect: "'invariant'" },
          { name: 'a circular convolution commutes with shifts', setup: 'roll = lambda v: v[-1:] + v[:-1]\nconv = lambda v: [v[i - 1] - 2 * v[i] + v[(i + 1) % len(v)] for i in range(len(v))]', call: 'symmetry_check(conv, roll, [[0, 1, 4, 2, 0, 0], [3, -1, 2, 5, 1, 0]])', expect: "'equivariant'" },
          { name: 'a layer with weights that depend on position', setup: 'roll = lambda v: v[-1:] + v[:-1]\nscale = lambda v: [x * (i + 1) for i, x in enumerate(v)]', call: 'symmetry_check(scale, roll, [[1, 2, 3, 4]])', expect: "'neither'" },
          { name: 'sorting ignores order', call: 'symmetry_check(sorted, lambda v: v[::-1], [[3, 1, 2], [5, 5, 0, 7]])', expect: "'invariant'" },
          { name: 'the tolerance matters', setup: 'roll = lambda v: v[-1:] + v[:-1]\nnoisy = lambda v: [sum(v) + 1e-6 * v[0]]', call: '(symmetry_check(noisy, roll, [[1.0, 2.0, 3.0]]), symmetry_check(noisy, roll, [[1.0, 2.0, 3.0]], tol=1e-3))', expect: "('neither', 'invariant')" },
          { name: 'random checks vs reference', setup: 'import random\nrandom.seed(5)\nroll = lambda v: v[-1:] + v[:-1]\nW = [[random.gauss(0, 1) for _ in range(5)] for _ in range(5)]\ndense = lambda v: [sum(w * x for w, x in zip(row, v)) for row in W]\nmx = lambda v: [max(v)]\nxs = [[random.gauss(0, 1) for _ in range(5)] for _ in range(3)]', call: '(symmetry_check(dense, roll, xs), symmetry_check(mx, roll, xs), symmetry_check(lambda v: [2 * x for x in v], roll, xs))', hidden: true }
        ]
      },
      {
        id: 'same-shape', title: 'Same Shape, Any Pose', difficulty: 'medium',
        prompt: `Two point sets describe the same rigid shape if one can be rotated, shifted and reordered into the other. A simple invariant test compares their **sorted lists of pairwise distances**. Implement \`same_shape(a, b, tol=1e-6)\` that returns True when the two sets have the same number of points and their sorted pairwise distances (over all pairs i < j) agree to within \`tol\`.

(Different shapes can occasionally share all distances, so this is a test, not a proof, but it is a good, fast invariant.)`,
        starter: `
import math

def same_shape(a, b, tol=1e-6):
    # TODO
    pass
`,
        solution: `
import math

def same_shape(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    sig = lambda pts: sorted(math.dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))
    return all(abs(x - y) <= tol for x, y in zip(sig(a), sig(b)))
`,
        hints: ['Compute the distances for all pairs i < j and sort them.', 'Sets of different sizes can never match.'],
        explain: `Sorting removes the order of the points, distances remove the pose. Models that only see invariant features can never be fooled by a rotated copy, which is exactly the guarantee you want for molecular properties.`,
        tests: [
          { name: 'a shifted copy', call: 'same_shape([(0, 0), (1, 0), (0, 2)], [(5, 5), (6, 5), (5, 7)])', expect: 'True' },
          { name: 'a rotated and reordered copy', call: 'same_shape([(0, 0), (1, 0), (0, 2)], [(0, 1), (0, 0), (-2, 0)])', expect: 'True' },
          { name: 'a scaled copy is a different shape', call: 'same_shape([(0, 0), (1, 0), (0, 2)], [(0, 0), (2, 0), (0, 4)])', expect: 'False' },
          { name: 'different sizes', call: 'same_shape([(0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)])', expect: 'False' },
          { name: 'a random rotation in 3D', setup: 'import numpy as np\nrng = np.random.default_rng(3)\nP = rng.normal(size=(6, 3))\nq, r = np.linalg.qr(rng.normal(size=(3, 3)))\nQ = q * np.sign(np.diag(r))\nP2 = (P @ Q.T + 1.5)[::-1]', call: 'same_shape(P.tolist(), P2.tolist())', expect: 'True', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Bronstein et al., Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges (2021)', url: 'https://arxiv.org/abs/2104.13478' },
      { title: 'Satorras et al., E(n) Equivariant Graph Neural Networks (2021)', url: 'https://arxiv.org/abs/2102.09844' },
      { title: 'Zaheer et al., Deep Sets (2017)', url: 'https://arxiv.org/abs/1703.06114' }
    ]
  },

  // =====================================================================================
  {
    id: 'interpretability-practice', branch: 'spec', title: 'Mechanistic Interpretability in Practice', short: 'Interpretability in Practice', level: 3, minutes: 45, prereqs: ['interpretability'],
    tags: ['mechanistic interpretability', 'superposition', 'toy models', 'sparse autoencoder', 'dictionary learning', 'features', 'monosemantic', 'dead latents', 'activation patching', 'causal tracing', 'circuits', 'transformerlens'],
    summary: 'Hands-on reverse engineering of neural networks: training a toy model that packs more features than it has dimensions (superposition), training a sparse autoencoder that pulls those features apart again, and measuring the causal role of components with activation patching, plus the workflow, tools and limits of the field.',
    why: `The Interpretability lesson explains the ideas; this one has you run them. Seeing a five-dimensional layer store eighteen features, and then watching a sparse autoencoder recover hidden directions that PCA cannot find, makes superposition and dictionary learning concrete. These are the same techniques used to find millions of interpretable features inside frontier language models, and they are increasingly part of safety and debugging work.`,
    objectives: [
      'Train a toy model and observe superposition as features become sparse',
      'Train a sparse autoencoder and check which hidden features it recovers',
      'Compute and interpret activation-patching effects',
      'Describe the workflow, tools and limitations of mechanistic interpretability'
    ],
    sections: [
      { type: 'text', title: "A working method", md: `Mechanistic interpretability tries to explain how a network computes something, in terms of internal features and the circuits that connect them (the ideas are introduced in [Interpretability](#/concept/interpretability)). Most projects follow the same loop:

1. **Pick a narrow behaviour and build a dataset for it**: prompts where the model shows it, and closely matched prompts where it does not. "Completes *When Mary and John went to the store, John gave a drink to* with *Mary*" is a better target than "understands grammar".
2. **Localize.** Find the layers, token positions and components that matter, with activation patching or ablations (switching a component off).
3. **Describe what is represented there.** Look at attention patterns, train linear probes, or decompose the activations into features with a sparse autoencoder.
4. **Test the explanation causally.** If a feature or attention head is claimed to carry the behaviour, removing or amplifying it should change the behaviour as predicted, on prompts that were not used to find it.

The common tools: **TransformerLens** wraps GPT-style models so that every activation has a named hook you can read or overwrite; **nnsight** does the same for any PyTorch model and can run interventions on large open models hosted remotely; **SAELens** trains and loads sparse autoencoders, and **Neuronpedia** is a website for browsing their features. Open sparse autoencoders exist for several models, for example Google DeepMind's **Gemma Scope** (2024), which covers every layer of Gemma 2 2B and 9B.` },
      { type: 'text', title: "Superposition in a toy model", md: `A layer with m dimensions can hold m features in perpendicular directions without any interference. Anthropic's *Toy Models of Superposition* (2022) asked what happens when there are more features than dimensions. The model compresses n features into m < n dimensions and decompresses them with the same weights:

$$ h = W x,    x̂ = ReLU( Wᵀ h + b ),    loss = Σ_i I_i ( x_i − x̂_i )² $$
:::where
x — a vector of n feature values between 0 and 1; with sparsity s, each value is zero with probability s
W — an m × n weight matrix with m < n; column i is the direction that stores feature i
h — the m-dimensional hidden representation
b — a bias vector of length n
x̂ — the reconstructed features
I_i — the importance of feature i, which weights its error in the loss
:::

When features are dense, the best the model can do is store the m most important ones and drop the rest, much like PCA. When features are sparse, two features are rarely active at the same time, so storing them in overlapping directions causes little interference, and the ReLU with a negative bias filters out what remains. The code trains this model with 20 features and 5 dimensions at increasing sparsity and counts the features that get a direction of their own: a column of W with a length near 1 rather than near 0.` },
      { type: 'code', title: "Superposition appears as features become sparse", code: `import numpy as np
rng = np.random.default_rng(0)
n, m = 20, 5                                         # 20 features squeezed into 5 hidden dimensions
importance = 0.85 ** np.arange(n)                    # earlier features matter more to the loss

def train(sparsity, steps=1000, batch=256, lr=0.01):
    W, b = rng.normal(0, 0.3, (m, n)), np.zeros(n)
    mW, vW, mb, vb = 0 * W, 0 * W, 0 * b, 0 * b
    for k in range(1, steps + 1):
        X = rng.random((batch, n)) * (rng.random((batch, n)) > sparsity)    # each feature is zero with probability \`sparsity\`
        H = X @ W.T                                  # compress: 20 numbers -> 5
        Z = H @ W + b                                # decompress with the same weights
        A = np.maximum(Z, 0)
        G = 2 * importance * (A - X) * (Z > 0) / batch                       # gradient of the weighted squared error
        gW = H.T @ G + (G @ W.T).T @ X
        gb = G.sum(0)
        for p, g, mom, vel in ((W, gW, mW, vW), (b, gb, mb, vb)):            # Adam updates
            mom *= 0.9; mom += 0.1 * g
            vel *= 0.999; vel += 0.001 * g * g
            p -= lr * (mom / (1 - 0.9 ** k)) / (np.sqrt(vel / (1 - 0.999 ** k)) + 1e-8)
    return W

print("sparsity   features represented (of 20) in 5 dimensions")
counts = {}
for s in (0.0, 0.7, 0.9, 0.97, 0.99):
    W = train(s)
    counts[s] = int(np.sum(np.linalg.norm(W, axis=0) > 0.5))                # a feature is represented if its column is long
    print(f"{s:8.2f} {counts[s]:10d}")
print("sparse features are packed into superposition:", counts[0.99] > 2 * counts[0.0])`, after: `With dense features, the model keeps the five most important ones, one per dimension, and drops the other fifteen. At moderate sparsity it stores ten, as five pairs of features pointing in exactly opposite directions: when one of a pair is active, the ReLU removes its negative echo on the other. At 99% sparsity, 18 of the 20 features share the five dimensions. Language models appear to do the same with far more features than neurons, which is why single neurons are often polysemantic and why tools such as sparse autoencoders are needed to pull the features apart.`, predict: {"type":"mcq","q":"As the features become sparser, how many of the 20 features will the 5-dimensional layer represent?","options":["Always 5, one per dimension","Fewer, because sparse features carry less information","More, because many features can share the 5 dimensions","All 20, at every sparsity"],"answer":2,"explain":"Sparse features are rarely active together, so sharing directions costs little: the count rises from 5 with dense features to 18 at 99% sparsity.","expect":["sparse features are packed into superposition: True"]} },
      { type: 'chart', title: "How long is each feature's direction?", md: `The lengths of the 20 columns of W in two of the models trained above:`, after: `With dense features, the five most important features get directions of length 1 and the other fifteen stay close to 0: the model ignores them. At 99% sparsity, the 16 most important features have full-length directions and two more have about half, all within five dimensions. In both cases, the features left out are the least important ones.`, chart: {"kind":"bar","x":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"],"series":[{"name":"dense features (sparsity 0)","y":[0.999,1,1.001,0.999,0.998,0.06,0.034,0.013,0.011,0.016,0.014,0.008,0.014,0.006,0.008,0.005,0.008,0.004,0.004,0.004]},{"name":"sparse features (sparsity 0.99)","y":[1.154,1.175,1.148,1.169,1.148,1.125,1.138,1.136,1.117,1.131,1.097,1.148,1.094,1.118,1.081,1.088,0.54,0.529,0.036,0.076]}],"xLabel":"feature (1 is the most important)","yLabel":"length of its column in W","yMin":0,"hlines":[{"y":0.5,"label":"counted as represented"}]} },
      { type: 'text', title: "Sparse autoencoders", md: `Superposition makes neurons the wrong unit of analysis: each neuron mixes many features, and each feature is spread over many neurons. A **sparse autoencoder** (SAE) tries to undo this. It is trained on a model's activations to rebuild each one as a sum of a few directions chosen from a large dictionary:

$$ c = ReLU( W_e ( a − b_d ) + b_e ),    â = W_d c + b_d,    loss = ‖ a − â ‖² + λ ‖ c ‖₁ $$
:::where
a — an activation vector from the model, for example the residual stream at one layer for one token
W_e, b_e — the encoder's weights and bias
c — the code: one non-negative value per latent (dictionary entry), most of them zero
W_d, b_d — the decoder's weights and bias; each column of W_d is a learned feature direction, kept at unit length
â — the reconstruction of the activation
λ — the strength of the sparsity penalty
‖ c ‖₁ — the sum of the code's values, which pushes most of them to exactly zero
:::

The dictionary is **overcomplete**: it has more latents than the activation has dimensions, often 10 to 100 times more. In practice:

- **Scale.** Anthropic trained SAEs with up to about 34 million latents on Claude 3 Sonnet (*Scaling Monosemanticity*, 2024) and found features for concepts such as the Golden Gate Bridge, bugs in code and deception, many of which changed the model's behaviour when amplified. OpenAI trained a 16-million-latent SAE on GPT-4 the same year.
- **Variants.** TopK SAEs keep exactly the k largest codes instead of using an L1 penalty, and JumpReLU SAEs (used for Gemma Scope) learn a threshold for each latent. Both avoid the L1 penalty's habit of shrinking the codes that stay active.
- **Dead latents** never activate and waste capacity. Training recipes resample them or add an auxiliary loss that gives them a gradient.
- **Evaluation** has two sides: how much of the model's performance survives when its activations are replaced by the reconstructions, and how interpretable the latents are, judged by people or by a language model that reads their top-activating examples.` },
      { type: 'code', title: "A sparse autoencoder finds features that PCA cannot", md: `Twenty hidden feature directions in a 10-dimensional space, each input mixing about two of them, as in a layer that uses superposition. A sparse autoencoder with 40 latents is trained on these activations, with its gradients written out by hand, and compared with PCA:`, code: `import numpy as np
rng = np.random.default_rng(0)
d, k_true, latents = 10, 20, 40                      # 20 hidden features in a 10-dimensional space; a dictionary of 40 latents
F = rng.normal(size=(k_true, d)); F /= np.linalg.norm(F, axis=1, keepdims=True)
def sample(batch):                                   # each activation mixes about 2 of the 20 features
    on = rng.random((batch, k_true)) < 2 / k_true
    return (on * rng.uniform(0.5, 1.5, (batch, k_true))) @ F

Wd = rng.normal(size=(d, latents)); Wd /= np.linalg.norm(Wd, axis=0, keepdims=True)   # decoder: one unit-length direction per latent
We, be, bd = Wd.T.copy(), np.zeros(latents), np.zeros(d)                              # the encoder starts as the decoder's transpose
lam, lr = 0.4, 0.02                                  # strength of the sparsity penalty, learning rate
params = [We, be, Wd, bd]
mom = [np.zeros_like(p) for p in params]; vel = [np.zeros_like(p) for p in params]
for step in range(1, 1501):
    X = sample(128)
    Xc = X - bd
    pre = Xc @ We.T + be
    C = np.maximum(pre, 0)                           # the codes: most latents are exactly zero
    Xh = C @ Wd.T + bd                               # the reconstruction
    dXh = 2 * (Xh - X) / len(X)                      # gradients of the squared error plus lam * (sum of the codes)
    dC = dXh @ Wd + lam * (C > 0) / len(X)
    dpre = dC * (pre > 0)
    grads = [dpre.T @ Xc, dpre.sum(0), dXh.T @ C, dXh.sum(0) - (dpre @ We).sum(0)]
    for i, (p, g) in enumerate(zip(params, grads)):  # Adam updates
        mom[i] = 0.9 * mom[i] + 0.1 * g
        vel[i] = 0.999 * vel[i] + 0.001 * g * g
        p -= lr * (mom[i] / (1 - 0.9 ** step)) / (np.sqrt(vel[i] / (1 - 0.999 ** step)) + 1e-8)
    Wd /= np.maximum(np.linalg.norm(Wd, axis=0, keepdims=True), 1e-8)                  # keep decoder directions unit length

X = sample(3000)
C = np.maximum((X - bd) @ We.T + be, 0)
print(f"reconstruction error {np.linalg.norm(C @ Wd.T + bd - X) / np.linalg.norm(X):.3f}, active latents per input {np.mean((C > 1e-3).sum(1)):.1f}")
found = int(np.sum(np.abs(F @ Wd).max(axis=1) > 0.9))           # best cosine similarity of each true feature with a learned direction
_, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
pca_found = int(np.sum(np.abs(F @ Vt.T).max(axis=1) > 0.9))
print(f"true features recovered (cosine > 0.9): sparse autoencoder {found} of {k_true}, PCA {pca_found} of {k_true}")
print("the sparse autoencoder finds most features, PCA almost none:", found > k_true // 2 and pca_found <= 2)`, after: `PCA can return at most 10 perpendicular directions, and the 20 features are neither that few nor perpendicular, so every principal component mixes several features and none matches one. The sparse autoencoder, free to use 40 directions as long as each input activates only a few, lines up 15 of its directions with true features. It is not perfect: the codes use about 5 latents per input where about 2 features are present, and the reconstruction misses part of the signal (a relative error of about 0.31). Raising λ makes the codes sparser and the reconstruction worse; lowering it does the opposite. SAEs on real models face the same trade-off, with millions of latents and billions of training activations.`, predict: {"type":"mcq","q":"Which method will find more of the 20 hidden feature directions?","options":["PCA, because it finds the directions of largest variance","The sparse autoencoder","Both find all 20","Neither finds any"],"answer":1,"explain":"PCA is limited to 10 perpendicular directions in a 10-dimensional space; the sparse autoencoder can use many more directions, one per feature.","expect":["the sparse autoencoder finds most features, PCA almost none: True"]} },
      { type: 'text', title: "Activation patching", md: `Activation patching (also called causal tracing or interchange intervention) asks whether one component carries the information that separates two inputs. Run the model on a **clean** prompt and on a **corrupted** prompt that differs in one detail, record a metric for both, then rerun the corrupted prompt with one activation copied in from the clean run:

$$ effect = ( m_patched − m_corrupted ) / ( m_clean − m_corrupted ) $$
:::where
m_clean — the metric on the clean prompt, for example the logit of the correct answer minus the logit of a wrong one
m_corrupted — the same metric on the corrupted prompt
m_patched — the metric on the corrupted prompt with one activation (a layer, a position, a head or a feature) copied from the clean run
effect — the share of the clean behaviour that this one activation restores: 1 for all of it, 0 for none
:::

Sweeping the patch over every layer and token position gives a map of where the information sits. In the example above, it sits at the position of the repeated name in early layers and moves to the final position in later layers, carried there by specific attention heads (Wang et al., 2022). Three choices change the answer:

- **The corruption.** Swapping one name for another (a minimal pair) gives cleaner results than adding noise to the embeddings, and a logit difference is a better metric than a probability, which saturates.
- **The direction.** Patching clean activations into the corrupted run (denoising) finds components that are *sufficient* to restore the behaviour; patching corrupted activations into the clean run (noising) finds components that are *necessary*. They can disagree when several components back each other up.
- **The cost.** One forward pass per patch adds up for large models. **Attribution patching** estimates all patches at once from gradients, as a first pass before exact patching.` },
      { type: 'code', title: "Activation patching with TransformerLens", run: false, md: `Run this on your own computer after \`pip install transformer_lens\`; it downloads GPT-2 small (about 550 MB). Each patch copies the clean residual stream at one layer and one token position into the corrupted run:`, code: `# pip install transformer_lens        (run on your own computer)
from functools import partial
import torch
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2-small")
clean = model.to_tokens("When Mary and John went to the store, John gave a drink to")
corrupted = model.to_tokens("When Mary and John went to the store, Mary gave a drink to")
mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")

def logit_diff(logits):                     # how strongly the model prefers " Mary" over " John" as the next token
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

clean_logits, cache = model.run_with_cache(clean)
m_clean, m_corrupted = logit_diff(clean_logits), logit_diff(model(corrupted))

def patch(resid, hook, pos):                # overwrite one position of the residual stream with its clean value
    resid[:, pos] = cache[hook.name][:, pos]
    return resid

effect = torch.zeros(model.cfg.n_layers, clean.shape[1])
for layer in range(model.cfg.n_layers):
    for pos in range(clean.shape[1]):
        hook_name = utils.get_act_name("resid_pre", layer)
        logits = model.run_with_hooks(corrupted, fwd_hooks=[(hook_name, partial(patch, pos=pos))])
        effect[layer, pos] = (logit_diff(logits) - m_corrupted) / (m_clean - m_corrupted)
print(f"clean {m_clean:.2f}, corrupted {m_corrupted:.2f}")
print(effect.round(decimals=2))             # rows are layers, columns token positions; values near 1 restore the answer`, after: `The two prompts differ only in the repeated name, so they have the same number of tokens and every position can be patched. Plotting \`effect\` as a heat map shows the pattern described above, and the same loop with \`hook_z\` (the output of each attention head) instead of \`resid_pre\` narrows it down to individual heads.` },
      { type: 'text', title: "Limits and good practice", md: `- **Features are a model of the model.** An SAE's latents depend on its size, its sparsity penalty and its training data. Larger dictionaries split one feature into several finer ones (feature splitting), and the part of an activation that an SAE fails to reconstruct can still matter for the model's behaviour.
- **Labels from top examples can mislead.** A latent that fires strongly on French text may also fire weakly on Spanish. Check the whole range of activations, and test the label by steering or ablating the latent.
- **Compare with simple baselines.** For specific practical tasks, such as detecting harmful requests, some studies have found that linear probes trained on the raw activations match or beat methods based on SAE features.
- **Toy models are not language models.** Results from a toy setup are hypotheses about a real model, not conclusions.
- **Newer methods** replace MLP layers with sparse, interpretable approximations (transcoders and cross-layer transcoders) and trace how features feed into each other on a single prompt, producing attribution graphs (Anthropic, 2025). They are promising, but they still explain only part of what large models compute.` },
      { type: 'check', question: {"type":"mcq","q":"An SAE latent's 20 strongest activations are all on text about cooking. What should you do before calling it a cooking feature?","options":["Nothing, 20 examples are enough","Check its weaker activations too, and test whether amplifying or removing it changes cooking-related outputs","Retrain the language model","Count how many latents the SAE has"],"answer":1,"explain":"Top examples show only the strongest cases. A label should hold across the range of activations and survive a causal test."} }
    ],
    when: `- Debugging why a model behaves in a specific way, beyond input attributions.
- Safety work: finding internal features related to deception, bias or dangerous knowledge, and checking whether training changed them.
- Research on how capabilities are implemented, from toy models to large language models.`,
    pitfalls: `- Reading meaning into a feature from a few top examples: check many activations, including weak ones.
- Treating a good reconstruction as proof of understanding: an autoencoder can reconstruct well with features that are hard to interpret.
- Patching without a careful corrupted baseline: the effect depends on what you patch in.
- Generalizing from toy models to large ones without checking.`,
    takeaways: [
      'When features are sparse, a layer can store more features than it has dimensions, at the cost of some interference: superposition.',
      'Sparse autoencoders learn an overcomplete dictionary in which each input uses only a few features, recovering directions that PCA cannot.',
      'Activation patching measures how much one component\'s activation carries the difference between a clean and a corrupted input.',
      'Interpretability claims should be tested causally, by intervening on the model, not only by correlation.'
    ],
    quiz: [
      {"type":"mcq","q":"In the toy model, what changes when the features become sparse?","options":["The model represents fewer features","The model represents more features than it has dimensions, in overlapping directions","The loss becomes exactly zero","Nothing changes"],"answer":1,"explain":"Sparse features are rarely active together, so sharing directions causes little interference: 5 dimensions held 18 of 20 features at 99% sparsity."},
      {"type":"tf","q":"With dense features, the toy model keeps the most important features, one per dimension, and drops the rest.","answer":true,"explain":"Without sparsity, overlapping features would interfere all the time, so the best solution resembles PCA: the 5 most important features, each in its own direction."},
      {"type":"mcq","q":"What is the purpose of the penalty λ‖c‖₁ in a sparse autoencoder's loss?","options":["It makes the reconstruction exact","It pushes most code values to zero, so each input is explained by a few latents","It keeps the decoder columns at unit length","It speeds up training"],"answer":1,"explain":"Sparsity is what makes latents line up with individual features instead of mixtures. TopK and JumpReLU SAEs reach it in other ways."},
      {"type":"mcq","q":"Why does PCA fail to recover the hidden features in the code example?","options":["PCA needs more data","PCA gives at most as many perpendicular directions as there are dimensions, while the features are more numerous and not perpendicular","PCA cannot handle negative values","PCA is too slow"],"answer":1,"explain":"Ten perpendicular directions cannot match twenty features that overlap, so each principal component mixes several of them."},
      {"type":"fill","q":"The logit difference is 3.0 on the clean prompt and −1.0 on the corrupted one. Patching one head into the corrupted run gives 2.0. What is the patching effect?","answer":["0.75",".75","3/4"],"explain":"(2.0 − (−1.0)) / (3.0 − (−1.0)) = 3 / 4 = 0.75: the head restores three quarters of the clean behaviour."},
      {"type":"mcq","q":"Patching clean activations into the corrupted run (denoising) shows that a component is…","options":["necessary for the behaviour","sufficient to restore the behaviour","both necessary and sufficient","irrelevant to the behaviour"],"answer":1,"explain":"Denoising finds components whose clean values are enough to restore the behaviour; noising, which patches corrupted values into the clean run, finds the components the behaviour needs."}
    ],
    problems: [
      {
        id: 'patching-effect', title: 'Activation Patching Effect', difficulty: 'easy',
        prompt: `In activation patching you run a model on a **clean** input and a **corrupted** input, then rerun the corrupted input with one component's activation copied from the clean run. Implement \`patching_effect(clean, corrupted, patched)\`, where the three arguments are the model's output scores (for example the logit of the correct answer) in the three runs:

effect = (patched − corrupted) / (clean − corrupted)

Return 0.0 when clean equals corrupted. An effect of 1 means the patch restored the clean behaviour completely; 0 means it changed nothing.`,
        starter: `
def patching_effect(clean, corrupted, patched):
    # TODO
    pass
`,
        solution: `
def patching_effect(clean, corrupted, patched):
    if clean == corrupted:
        return 0.0
    return (patched - corrupted) / (clean - corrupted)
`,
        hints: ['Normalize by the gap between the clean and the corrupted output.'],
        explain: `Normalizing makes effects comparable across components and inputs. Sweeping this over every layer and position produces the heat maps used to localize where a model stores a fact or computes a step.`,
        tests: [
          { name: 'full restoration', call: 'patching_effect(4.0, 1.0, 4.0)', expect: '1.0' },
          { name: 'no effect', call: 'patching_effect(4.0, 1.0, 1.0)', expect: '0.0' },
          { name: 'partial', call: 'patching_effect(4.0, 1.0, 2.5)', expect: '0.5' },
          { name: 'no gap to explain', call: 'patching_effect(2.0, 2.0, 3.0)', expect: '0.0' },
          { name: 'a negative effect', call: 'patching_effect(3.0, 1.0, 0.0)', hidden: true }
        ]
      },
      {
        id: 'features-recovered', title: 'How Many Features Were Recovered?', difficulty: 'medium',
        prompt: `To evaluate a sparse autoencoder on synthetic data, compare its learned dictionary directions with the true feature directions. Implement \`count_recovered(true_dirs, learned_dirs, threshold=0.9)\`: both arguments are lists of vectors (not necessarily of unit length). A true direction counts as recovered when the **absolute cosine similarity** with at least one learned direction is above \`threshold\`. Return the number of recovered true directions.`,
        starter: `
import math

def count_recovered(true_dirs, learned_dirs, threshold=0.9):
    # TODO
    pass
`,
        solution: `
import math

def count_recovered(true_dirs, learned_dirs, threshold=0.9):
    def cos(u, v):
        dot = sum(a * b for a, b in zip(u, v))
        return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))
    return sum(1 for t in true_dirs if max(abs(cos(t, l)) for l in learned_dirs) > threshold)
`,
        hints: ['Cosine similarity: the dot product divided by both lengths.', 'Use the absolute value: a feature learned with the opposite sign still counts.'],
        explain: `This is how sparse autoencoders are checked on toy data where the true features are known. On real models the true features are unknown, so researchers instead examine what activates each learned feature and test its causal effect.`,
        tests: [
          { name: 'exact matches', call: 'count_recovered([[1, 0], [0, 1]], [[2, 0], [0, -3]])', expect: '2' },
          { name: 'a missing direction', call: 'count_recovered([[1, 0], [0, 1]], [[1, 0.1]])', expect: '1' },
          { name: 'below the threshold', call: 'count_recovered([[1, 0]], [[1, 1]])', expect: '0' },
          { name: 'random directions vs reference', setup: 'import random\nrandom.seed(17)\nT = [[random.gauss(0, 1) for _ in range(6)] for _ in range(8)]\nL = [[x + random.gauss(0, 0.2) for x in t] for t in T[:5]] + [[random.gauss(0, 1) for _ in range(6)] for _ in range(4)]', call: 'count_recovered(T, L)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Elhage et al., Toy Models of Superposition (Anthropic, 2022)', url: 'https://transformer-circuits.pub/2022/toy_model/index.html' },
      { title: 'Bricken et al., Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Anthropic, 2023)', url: 'https://transformer-circuits.pub/2023/monosemantic-features/index.html' },
      { title: 'TransformerLens, a library for mechanistic interpretability', url: 'https://github.com/TransformerLensOrg/TransformerLens' }
    ]
  },

  // =====================================================================================
  {
    id: 'ai-hardware', branch: 'llm', title: 'AI Hardware & Systems: GPUs, Memory and Compilers', short: 'AI Hardware & Systems', level: 3, minutes: 45, prereqs: ['training-at-scale', 'llm-serving'],
    tags: ['gpu', 'tpu', 'accelerator', 'tensor cores', 'streaming multiprocessor', 'warp', 'memory hierarchy', 'hbm', 'shared memory', 'tiling', 'kernel fusion', 'triton', 'torch.compile', 'cuda', 'nvlink', 'infiniband', 'precision', 'accumulation', 'energy'],
    summary: 'The machines that train and run modern AI: how GPUs organize thousands of threads and tensor cores, the memory hierarchy from registers to high-bandwidth memory, why tiling and data reuse decide speed, kernel fusion and compilers (Triton, torch.compile), number formats and why sums are accumulated in higher precision, interconnects between GPUs, and power and cost.',
    why: `The same model can run ten times faster or slower depending on how its computation meets the hardware. Most operations in deep learning are limited not by arithmetic but by moving data, and the tricks that fix this (tiling, fusion, low precision with careful accumulation) explain FlashAttention, quantized inference and why training clusters are built the way they are. Knowing the hardware turns performance from folklore into arithmetic.`,
    objectives: [
      'Describe GPU execution (threads, warps, streaming multiprocessors, tensor cores) and the memory hierarchy',
      'Compute how tiling changes memory traffic and arithmetic intensity',
      'Explain kernel fusion and estimate its speed-up for memory-bound operations',
      'Explain why low-precision values are accumulated in higher precision, and what interconnects mean for scaling'
    ],
    sections: [
      { type: 'text', title: "Inside a GPU", md: `A GPU is a large array of simple processors built to run the same instruction on many values at once. Taking NVIDIA's H100 (2022) as the example:

- It has 132 **streaming multiprocessors** (SMs). Each SM runs many **threads**, scheduled in groups of 32 called **warps**; the threads of a warp execute the same instruction on different data.
- Programs for the GPU are called **kernels**. A kernel launches a grid of thread **blocks**; each block runs on one SM, and its threads can cooperate through the SM's fast shared memory.
- Each SM has four **tensor cores**, units that perform a small matrix multiply-accumulate in one instruction. They supply almost all of the chip's speed for deep learning: about 990 × 10¹² operations per second (990 TFLOP/s) on 16-bit inputs, against about 67 TFLOP/s for ordinary 32-bit arithmetic.
- Waiting for memory is hidden by switching: while one warp waits for data, the SM runs another, so a kernel needs many warps in flight to keep the hardware busy.

Other accelerators make different trade-offs. Google's **TPUs** are built around systolic arrays, grids of multiply-accumulate cells through which matrix tiles flow in lockstep, and connect thousands of chips directly to each other in pods. AMD's Instinct GPUs offer more memory per chip, and designs such as Cerebras's wafer-scale engine and Groq's LPU hold model weights in large amounts of on-chip memory to avoid slow external memory.` },
      { type: 'text', title: "The memory hierarchy", md: `Data reaches the tensor cores through a hierarchy of memories that get larger and slower the further they are from the arithmetic. Approximate figures for an H100:

| Memory | Size | Shared by | Speed |
|---|---|---|---|
| Registers | 256 KB per SM | one thread | the fastest |
| Shared memory and L1 cache | up to 228 KB of shared memory per SM | the threads of one block | tens of TB/s across the chip |
| L2 cache | 50 MB | all SMs | several TB/s |
| HBM (main memory) | 80 GB | the whole GPU | 3.35 TB/s |

Beyond one GPU, links get slower again: about 900 GB/s in total over NVLink to the other GPUs in a server, about 64 GB/s each way over PCIe to the CPU, and about 50 GB/s each way per 400 Gb/s network link between servers.

This explains the central fact of GPU performance: arithmetic is cheap and moving data is expensive. A kernel's time is set by whichever takes longer, its arithmetic or its memory traffic (the roofline model from [LLM Serving](#/concept/llm-serving)):

$$ time ≈ max( FLOPs / peak ,  bytes / bandwidth ),    ridge point = peak / bandwidth $$
:::where
FLOPs — the floating-point operations the kernel performs
bytes — the data it reads from and writes to main memory
peak — the hardware's top arithmetic rate, about 990 × 10¹² FLOP/s in BF16 on an H100
bandwidth — the main-memory bandwidth, about 3.35 × 10¹² bytes per second on an H100
ridge point — the arithmetic intensity (FLOPs per byte) at which both take equally long: about 296 on an H100
:::

A kernel that performs fewer than about 296 operations per byte it moves is memory-bound, and faster arithmetic does not help it. Elementwise operations (around one operation per byte) and decoding with small batches sit far below that line. Large matrix multiplications can sit far above it, but only if the kernel reuses the data it loads.` },
      { type: 'text', title: "Tiling: reuse what you have loaded", md: `In C = A B, every output value needs a full row of A and a full column of B. Computed one output at a time, every multiply-add reads two values from main memory, which is half a floating-point operation per byte for 16-bit values. A tiled kernel instead loads a t × t block of A and a t × t block of B into shared memory and computes a t × t block of C from them, so every value it loads is used t times:

$$ values read ≈ 2n³ / t $$
:::where
n — the size of the two n × n matrices
t — the tile size: blocks of t × t values are loaded into shared memory and reused
values read — the values fetched from main memory; with t = 1 (no reuse), two for every multiply-add
:::` },
      { type: 'code', title: "How tiling raises arithmetic intensity", code: `n = 4096
flops = 2 * n ** 3                                   # multiply-adds of an n x n matrix product, counted as 2 operations each
def moved(tile):                                     # values read from main memory, in a simplified model
    return 2 * n ** 3 // tile + n ** 2               # each loaded tile is reused \`tile\` times; the result is written once
ridge = 990e12 / 3.35e12                             # a high-end GPU: about 990 BF16 TFLOP/s and 3.35 TB/s of memory bandwidth
print(f"the GPU needs about {ridge:.0f} FLOPs per byte moved to keep its arithmetic units busy")
print("tile size   values moved      arithmetic intensity (FLOPs per byte, 2-byte values)")
for t in (1, 8, 32, 64, 128, 256):
    print(f"{t:9d} {moved(t):16,} {flops / (2 * moved(t)):12.1f}")
print("64 x 64 tiles move about 64 times less data:", 60 < moved(1) / moved(64) < 65)`, after: `Without reuse, the product does half an operation per byte and would run at well under 1% of the GPU's peak. Doubling the tile size halves the traffic, and 64 × 64 tiles cut it by a factor of 64. In this simple count, tiles would need to be about 640 values wide to pass the ridge point. Real kernels get there by combining tiles in shared memory and registers with reuse in the L2 cache, which is why large matrix multiplications often run at 60 to 80% of peak. FlashAttention applies the same idea to attention (see [Efficient Attention](#/concept/efficient-attention)).`, predict: {"type":"mcq","q":"With 64 × 64 tiles, roughly how much less data does the matrix product read than without tiling?","options":["About 2 times less","About 8 times less","About 64 times less","No less: the same values are needed"],"answer":2,"explain":"Every value loaded into a 64 × 64 tile is used 64 times, so the traffic falls by a factor of about 64.","expect":["64 x 64 tiles move about 64 times less data: True"]} },
      { type: 'chart', title: "Arithmetic intensity against tile size", md: `The simple count from the code above, from no reuse up to tiles as large as the matrices:`, after: `Intensity grows in proportion to the tile size and crosses the ridge point at tiles about 640 values wide. It cannot go past about 1,365 FLOPs per byte for these matrices, the point where every input value is read only once. Real kernels use tiles of 64 to 256 values in shared memory and get the rest of the reuse from the L2 cache.`, chart: {"kind":"line","x":[1,2,4,8,16,32,64,128,256,512,1024,2048,4096],"xLog":true,"yLog":true,"series":[{"name":"4096 × 4096 matrix product, 16-bit values","y":[0.4999,0.9998,1.999,3.996,7.984,15.94,31.75,63.02,124.1,240.9,455.1,819.2,1365]}],"xLabel":"tile size t (log scale)","yLabel":"FLOPs per byte moved (log scale)","hlines":[{"y":296,"label":"ridge point of an H100, about 296"}]} },
      { type: 'text', title: "Kernel fusion and compilers", md: `A model runs as a sequence of kernels. If each one reads its inputs from main memory and writes its result back, a chain of cheap elementwise operations (adding a bias, an activation function, dropout, a residual connection) spends nearly all its time moving data. **Fusion** merges such a chain into a single kernel that reads the inputs once, keeps intermediate values in registers and writes only the final result.

Several tools produce fused kernels:

- **torch.compile** (PyTorch 2.0, 2023) captures a model's operations as a graph and generates fused kernels, by default through its TorchInductor backend, which writes Triton code for GPUs.
- **XLA**, the compiler behind JAX and TPUs, fuses operations across the whole program.
- **Triton** is a Python-based language in which you write a kernel in terms of blocks of data, while the compiler handles threads, shared memory and tensor cores. Many custom kernels are now written in it rather than in CUDA C++.
- **Hand-written kernels** in CUDA, often built on NVIDIA's CUTLASS templates, remain the choice for the most important operations, such as FlashAttention, which fuses all of attention into one pass.
- **CUDA graphs** record a sequence of kernel launches and replay it with one call, which removes the CPU overhead of launching hundreds of small kernels. This matters most when decoding with small batches.` },
      { type: 'code', title: "What fusion saves", code: `n = 4096 * 4096                                      # elements of one activation tensor
bytes_per = 2                                        # BF16
# four elementwise steps: add bias, GELU, dropout, add the residual
unfused = (2 + 2 + 2 + 3) * n * bytes_per            # each step reads its input(s) and writes its output to main memory
fused = 3 * n * bytes_per                            # one fused kernel reads the input and the residual once and writes once
bandwidth = 3.35e12
for name, b in (("four separate kernels", unfused), ("one fused kernel", fused)):
    print(f"{name:22s} moves {b / 1e6:6.0f} MB  ->  about {b / bandwidth * 1e6:5.0f} microseconds at 3.35 TB/s")
flops = 20 * n
print(f"the arithmetic itself: about {flops / 990e12 * 1e6:.2f} microseconds at 990 TFLOP/s")
print("fusion makes it about three times faster:", 2.5 < unfused / fused < 3.5)`, after: `The four separate kernels move three times as many bytes as the fused one, and because the arithmetic takes well under a microsecond, the time follows the bytes: about 90 against 30 microseconds. The practice problem *Memory Traffic With and Without Fusion* generalizes this count. A transformer layer contains many such chains, which is one reason torch.compile speeds up many models by tens of percent without any change to their code.`, predict: {"type":"mcq","q":"As four separate kernels, these steps take about 90 microseconds. About how long will one fused kernel take?","options":["About 90 microseconds, because the arithmetic is the same","About 30 microseconds","About 1 microsecond","Longer, because the kernel is more complex"],"answer":1,"explain":"The operations are memory-bound, so time follows bytes moved: 9 memory accesses per element become 3.","expect":["fusion makes it about three times faster: True"]} },
      { type: 'text', title: "Number formats and accumulation", md: `Fewer bits per value mean less memory, less traffic and faster tensor cores: each halving of the bits, from 16 to 8 and on the newest chips to 4, roughly doubles the peak rate. Formats differ in how they split their bits between range (the exponent) and precision (the mantissa); [Quantization](#/concept/quantization) and [Training at Scale](#/concept/training-at-scale) cover how they are used. The precision sets how far apart neighbouring numbers are:

$$ spacing(x) ≈ |x| · 2^(−p) $$
:::where
x — a number stored in the format
p — the number of mantissa (fraction) bits: 23 in float32, 10 in float16, 7 in bfloat16, 3 or 2 in the 8-bit formats
spacing(x) — the gap between x and the next representable number, to within a factor of 2; adding anything smaller than half of it to x changes nothing
:::

This matters for long sums. A matrix product adds thousands of products into each output value, and once the running total is large, small contributions fall below half the spacing and vanish. Tensor cores therefore multiply 16-bit or 8-bit inputs but accumulate in 32-bit precision, and optimizers keep a 32-bit master copy of the weights so that small updates are not rounded away. Even this needs care: DeepSeek-V3's technical report (December 2024) found that FP8 matrix products on its GPUs kept only about 14 bits of precision while accumulating, and it copied partial sums into 32-bit registers every 128 values to keep training accurate.` },
      { type: 'code', title: "Why sums are accumulated in higher precision", code: `import numpy as np
x16, x32 = np.float16(1.0), np.float32(1.0)
for _ in range(10_000):                              # add 0.0001 ten thousand times: the exact answer is 2.0
    x16 = np.float16(x16 + np.float16(0.0001))
    x32 = np.float32(x32 + np.float32(0.0001))
print(f"float16 running sum: {float(x16):.4f}   float32 running sum: {float(x32):.4f}   exact: 2.0000")
print(f"gap between neighbouring float16 numbers near 1: {float(np.spacing(np.float16(1.0))):.6f}")
print(f"gap between neighbouring float32 numbers near 1: {float(np.spacing(np.float32(1.0))):.2e}")
print("the float16 sum never moves from 1:", float(x16) == 1.0)`, after: `Near 1, neighbouring float16 numbers are about 0.001 apart, so 1 + 0.0001 rounds back to exactly 1, and ten thousand additions change nothing. Float32 has 13 more mantissa bits and ends within 0.0002 of the exact answer. Bfloat16 would do even worse than float16 here: it trades precision for range, and its spacing near 1 is about 0.008.`, predict: {"type":"mcq","q":"Starting from 1.0, 0.0001 is added ten thousand times in float16. What will the running sum be?","options":["Exactly 2.0","About 2.0, with a small rounding error","1.0, because every addition is rounded away","Infinity, because float16 overflows"],"answer":2,"explain":"Half the gap between float16 numbers near 1 is about 0.0005, so adding 0.0001 always rounds back to 1.","expect":["the float16 sum never moves from 1: True"]} },
      { type: 'text', title: "Interconnects, power and cost", md: `Training a large model spreads it over thousands of GPUs, and the links between them shape how the work is split (see [Training at Scale](#/concept/training-at-scale)):

- **Inside a server**, typically 8 GPUs are joined through NVLink switches, so any GPU can reach any other at about 900 GB/s. Racks such as NVIDIA's GB200 NVL72 (2024 to 2025) extend one NVLink domain to 72 GPUs.
- **Between servers**, InfiniBand or Ethernet with remote direct memory access (RDMA) gives each GPU a network link of 400 Gb/s or more, roughly a tenth of its NVLink bandwidth.
- **TPU pods** connect thousands of chips directly to their neighbours in a torus, with optical switches that can rewire the topology.
- **Parallelism follows the bandwidth.** Tensor parallelism, which exchanges activations inside every layer, stays within an NVLink domain; pipeline and data parallelism, which communicate less often, span the slower network.

Power and money follow the same physics:

- An H100 draws up to 700 W and the newer Blackwell GPUs about 1,000 W or more. A cluster of 100,000 GPUs needs well over 100 megawatts once servers, networking and cooling are included, and the densest racks are now liquid-cooled.
- Fetching a value from off-chip memory costs on the order of a hundred times more energy than an arithmetic operation on it, so tiling and fusion save energy as well as time.
- A training run costs roughly the number of GPUs times the hours times the price per GPU-hour, so every point of utilization (MFU) is money.` },
      { type: 'check', question: {"type":"mcq","q":"A layer-norm kernel performs about 5 FLOPs per byte it moves on an H100, whose ridge point is about 296. What will make it faster?","options":["Faster tensor cores","Moving fewer bytes, for example by fusing it with neighbouring operations or using a smaller number format","Performing more FLOPs per element","Nothing can help"],"answer":1,"explain":"At 5 FLOPs per byte the kernel is memory-bound: its time is set by the bytes it moves, not by its arithmetic."} }
    ],
    when: `- A model trains or serves more slowly than its FLOP count suggests.
- Choosing hardware, batch sizes, precisions or parallelism for a training or inference job.
- Reading about new kernels, accelerators or compiler features and judging what they change.`,
    pitfalls: `- Judging performance by FLOPs alone: memory traffic often decides the time.
- Accumulating long sums in 16-bit precision: small additions get rounded away.
- Optimizing a kernel before profiling: measure where the time goes first.
- Ignoring communication: past a few GPUs, moving data between them can dominate.`,
    takeaways: [
      'GPUs run thousands of threads in groups (warps) on streaming multiprocessors, with tensor cores for small matrix multiplications in low precision.',
      'Memory gets larger and slower from registers and shared memory to HBM; reusing data in fast memory is the key to speed.',
      'Tiling loads blocks into fast memory and reuses them, raising arithmetic intensity towards the hardware\'s ridge point.',
      'Fusing chains of elementwise operations into one kernel removes round trips to main memory, often for a several-fold speed-up.',
      'Low-precision formats save memory and time, but sums are accumulated in higher precision to avoid losing small contributions.'
    ],
    quiz: [
      {"type":"mcq","q":"On NVIDIA GPUs, how many threads make up a warp, the group that executes the same instruction together?","options":["8","32","128","1024"],"answer":1,"explain":"Threads are scheduled in warps of 32. When threads of one warp take different branches, the branches run one after the other, which wastes throughput."},
      {"type":"mcq","q":"Where does a tiled matrix-multiplication kernel keep the tiles it is reusing?","options":["In HBM","In shared memory and registers on each streaming multiprocessor","In the CPU's memory","On disk"],"answer":1,"explain":"The on-chip memories are small but much faster than HBM, so values loaded once can be reused many times there."},
      {"type":"tf","q":"Fusing a chain of elementwise operations into one kernel reduces the number of floating-point operations.","answer":false,"explain":"The arithmetic stays the same. Fusion removes the round trips of intermediate results to main memory, which is where the time went."},
      {"type":"fill","q":"A kernel performs 2 × 10⁹ FLOPs and moves 4 × 10⁹ bytes. What is its arithmetic intensity in FLOPs per byte?","answer":["0.5",".5","1/2"],"explain":"2 × 10⁹ / (4 × 10⁹) = 0.5 FLOPs per byte, far below an H100's ridge point of about 296, so the kernel is memory-bound."},
      {"type":"mcq","q":"Why do tensor cores accumulate the products of 16-bit inputs in 32-bit precision?","options":["32-bit arithmetic is faster","In a long sum, small contributions would be rounded away in 16-bit precision","The next layer requires 32-bit inputs","To save memory"],"answer":1,"explain":"Once the running total is large, a 16-bit format cannot represent the small increments, as the float16 example shows."},
      {"type":"mcq","q":"Which form of parallelism is usually kept inside one NVLink domain?","options":["Data parallelism","Tensor parallelism, which exchanges activations within every layer","Pipeline parallelism","None: all parallelism runs over the network"],"answer":1,"explain":"Tensor parallelism communicates in every layer of every step, so it needs the fastest links; data and pipeline parallelism communicate less often."}
    ],
    problems: [
      {
        id: 'bottleneck', title: 'Memory-Bound or Compute-Bound?', difficulty: 'easy',
        prompt: `A kernel performs \`flops\` floating-point operations and moves \`bytes_moved\` bytes to and from main memory. The hardware does at most \`peak_flops\` operations per second and moves at most \`bandwidth\` bytes per second. Implement \`bottleneck(flops, bytes_moved, peak_flops, bandwidth)\` returning \`"memory"\` if moving the data takes longer than the arithmetic, and \`"compute"\` otherwise.

\`\`\`text
bottleneck(2e9, 4e9, 1e15, 3e12)   ->  "memory"      0.5 FLOPs per byte, far below the ridge point
\`\`\``,
        starter: `
def bottleneck(flops, bytes_moved, peak_flops, bandwidth):
    # TODO
    pass
`,
        solution: `
def bottleneck(flops, bytes_moved, peak_flops, bandwidth):
    return "memory" if bytes_moved / bandwidth > flops / peak_flops else "compute"
`,
        hints: ['Compare the time to move the data with the time to do the arithmetic.', 'Equivalently, compare flops / bytes_moved with peak_flops / bandwidth (the ridge point).'],
        explain: `This is the roofline model in one line. Elementwise operations and decoding with small batches sit far below the ridge point, so making their arithmetic faster changes nothing; moving fewer bytes does.`,
        tests: [
          { call: 'bottleneck(2e9, 4e9, 1e15, 3e12)', expect: "'memory'" },
          { name: 'a large matrix multiplication', call: 'bottleneck(1.4e14, 1e11, 1e15, 3e12)', expect: "'compute'" },
          { name: 'exactly at the ridge counts as compute', call: 'bottleneck(300, 1, 300, 1)', expect: "'compute'" },
          { name: 'another kernel', call: 'bottleneck(5e12, 2e10, 9.9e14, 3.35e12)', hidden: true }
        ]
      },
      {
        id: 'fusion-traffic', title: 'Memory Traffic With and Without Fusion', difficulty: 'easy',
        prompt: `A chain of elementwise operations runs over every element of a tensor. \`inputs_per_op[i]\` is how many tensors operation i reads (its output always counts as one write). Count **memory accesses per element**:

- unfused: every operation reads all its inputs from main memory and writes its output, so each costs inputs + 1;
- fused into one kernel: the chain's first input is read once, every extra input (beyond the first of each operation) is read once, and only the final result is written.

Implement \`fusion_traffic(inputs_per_op)\` returning \`(unfused, fused)\`.

\`\`\`text
fusion_traffic([1, 1, 1, 2])   ->  (9, 3)      bias, GELU, dropout, then a residual add that reads one extra tensor
\`\`\``,
        starter: `
def fusion_traffic(inputs_per_op):
    # TODO
    pass
`,
        solution: `
def fusion_traffic(inputs_per_op):
    unfused = sum(k + 1 for k in inputs_per_op)
    fused = 1 + sum(k - 1 for k in inputs_per_op) + 1
    return unfused, fused
`,
        hints: ['Unfused: add up inputs + 1 for every operation.', 'Fused: one read of the first input, one read per extra input, one final write.'],
        explain: `For memory-bound chains, time is proportional to these counts, so fusing the four operations above runs about three times faster. Compilers such as torch.compile and XLA find many such fusions automatically, and kernels such as FlashAttention fuse far longer chains by hand.`,
        tests: [
          { call: 'fusion_traffic([1, 1, 1, 2])', expect: '(9, 3)' },
          { name: 'a single operation gains nothing', call: 'fusion_traffic([1])', expect: '(2, 2)' },
          { name: 'a longer chain', call: 'fusion_traffic([1] * 6)', expect: '(12, 2)' },
          { name: 'several extra inputs', call: 'fusion_traffic([2, 1, 3, 1])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Horace He, Making Deep Learning Go Brrrr From First Principles', url: 'https://horace.io/brrr_intro.html' },
      { title: 'Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)', url: 'https://arxiv.org/abs/2205.14135' },
      { title: 'NVIDIA, CUDA Programming Guide', url: 'https://docs.nvidia.com/cuda/cuda-programming-guide/' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
