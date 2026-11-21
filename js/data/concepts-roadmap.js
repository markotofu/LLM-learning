/* ML Atlas – content: lessons that complete the roadmap.sh AI Engineer and Machine Learning roadmaps:
 * AI engineering (models, APIs and the LLM stack), LLMs in production (observability and safety),
 * the Model Context Protocol, and discrete maths. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'ai-engineering', branch: 'apps', title: 'AI Engineering: Models, APIs and the LLM Stack', short: 'AI Engineering', level: 2, minutes: 45, prereqs: ['prompting'],
    tags: ['ai engineer', 'ml engineer', 'llm api', 'sdk', 'model selection', 'open-weight', 'closed models', 'claude', 'gpt', 'gemini', 'llama', 'qwen', 'mistral', 'deepseek', 'gemma', 'hugging face', 'ollama', 'lm studio', 'transformers.js', 'openrouter', 'openai-compatible', 'streaming', 'rate limits', 'retries', 'langchain', 'llamaindex', 'haystack', 'agent sdk', 'coding assistants'],
    summary: 'The practical stack for building products on pretrained models: what AI engineers do and how the role differs from ML engineering; choosing between closed and open-weight models by quality, cost, latency and privacy; calling models through APIs and SDKs (streaming, structured output, retries, OpenAI-compatible endpoints and routers); running open models locally; when frameworks such as LangChain and LlamaIndex help; and AI-assisted coding.',
    why: `Most teams that ship AI features never train a model. They choose one, call it through an API, and engineer everything around it: prompts, retrieval, tools, evaluation, cost and reliability. Those choices decide whether a product is fast, affordable and trustworthy, and they have to be revisited every few months as new models appear. This lesson maps that stack and the trade-offs behind each choice.`,
    objectives: [
      'Explain what an AI engineer does, and how the role differs from ML engineering and data science',
      'Choose a model by measuring quality, cost per outcome, latency, context length and privacy on your own task',
      'Call models robustly: streaming, structured output, retries with backoff, OpenAI-compatible endpoints and routers',
      'Run open-weight models locally, and decide when frameworks and agent SDKs are worth their abstraction'
    ],
    sections: [
      { type: 'text', title: "What an AI engineer does", md: `Three roles that are often confused:

| | AI engineer | ML engineer | Data scientist |
| --- | --- | --- | --- |
| Main question | How do we build a reliable product on top of existing models? | How do we train, deploy and operate models? | What does the data tell us, and which decisions should change? |
| Typical work | prompts and context, retrieval, agents and tools, evaluation, integration, cost and latency | data pipelines, training and fine-tuning, serving infrastructure, monitoring | analysis, experiments, statistical models, dashboards |
| Starts from | a pretrained model, behind an API or as open weights | data and a training objective | data and a business question |

The boundaries are fuzzy, and the skills overlap: an AI engineer who understands how models are trained (the rest of this atlas) makes better decisions about prompting, fine-tuning and evaluation.

**Impact on product development.** Features that used to need a research team (summarization, search over documents, assistants, extraction from messy text) can be prototyped in days. The hard part moves from *building a model* to *knowing whether its behaviour is good enough*: quality, safety and cost on real inputs. That is why teams write evaluation sets as specifications before they write prompts (see [Evaluating LLM Applications](#/concept/llm-evaluation)), and why product design has to plan for wrong answers: show sources, allow undo, hand over to a person when the model is unsure. On the engineering side, LLM features add streaming user interfaces, server-side handling of API keys, latency budgets and fallbacks when a provider is down.` },
      { type: 'text', title: "Choosing a model", md: `| | Closed (API) models | Open-weight models |
| --- | --- | --- |
| Examples | Anthropic's Claude, OpenAI's GPT and o-series, Google's Gemini; Mistral and Cohere also offer models through APIs | Meta's Llama, Mistral, Google's Gemma, Alibaba's Qwen, DeepSeek |
| Strengths | usually the most capable; nothing to host; tools, caching and long context built in | run anywhere; keep data in-house; fine-tune freely; the version never changes underneath you; no per-token fees |
| Costs | per-token prices; data leaves your systems (check the provider's retention and training terms); old versions are retired | hardware and operations; you handle scaling, safety filters and updates; licences vary, and some restrict use |

A sound selection process:

1. **Shortlist** with public benchmarks and leaderboards, the providers' documentation (context length, tools, modalities) and your constraints (privacy, region, budget).
2. **Evaluate on your own evaluation set**, never on leaderboard rank alone (see [Evaluating LLM Applications](#/concept/llm-evaluation)).
3. **Measure cost and latency at realistic volume**, and compare the cost of an *outcome*, not of a token (next code).
4. **Use tiers.** A small, fast model can classify or route requests, and a large or reasoning model can handle the hard ones.
5. **Pin versions** in production, re-run the evaluation when you change them, and keep a fallback model or provider.` },
      { type: 'code', title: "Cost per outcome, not per token", md: `Three hypothetical models with different prices and accuracies on your own evaluation set. A wrong answer is not free: it costs a human follow-up.`, code: `models = {   # hypothetical prices (dollars per million tokens) and accuracy measured on your own evaluation set
    "large":  {"price_in": 3.00, "price_out": 15.00, "accuracy": 0.92},
    "medium": {"price_in": 0.80, "price_out": 4.00, "accuracy": 0.85},
    "small":  {"price_in": 0.15, "price_out": 0.60, "accuracy": 0.70},
}
tokens_in, tokens_out = 1_500, 300            # a typical request
error_cost = 2.00                             # each wrong answer costs a human follow-up worth $2

print("model   cost per request   cost per correct answer   total per 1,000 requests incl. errors")
total = {}
for name, m in models.items():
    per_request = (tokens_in * m["price_in"] + tokens_out * m["price_out"]) / 1e6
    total[name] = 1000 * (per_request + (1 - m["accuracy"]) * error_cost)
    print(f"{name:7s} {per_request:13.5f} $ {per_request / m['accuracy']:20.5f} $ {total[name]:30.2f} $")
best = min(total, key=total.get)
print("cheapest once errors are counted:", best)
print("the cheapest model per token is not the cheapest overall:", best != "small")`, after: `The small model is more than 20 times cheaper per request, but it is wrong on 30% of requests, and at $2 per mistake its errors cost far more than the tokens: $600 per 1,000 requests against $169 for the large model. When errors are cheap (a user simply asks again) the order reverses, which the chart below shows. Always put a price on errors, even a rough one, before comparing models.`, predict: {"type":"mcq","q":"Every wrong answer costs $2 of human follow-up. Which model is cheapest overall?","options":["The small one: it is more than 20 times cheaper per request","The medium one, as a compromise","The large one: its fewer errors save more than its higher price","All three cost the same"],"answer":2,"explain":"Per 1,000 requests the large model costs about $9 in tokens plus $160 in errors, against $0.40 plus $600 for the small one.","expect":["cheapest once errors are counted: large"]} },
      { type: 'chart', title: "Total cost per 1,000 requests as errors get more expensive", md: `The three models from the code, for error costs from nothing to 30 cents:`, after: `When errors cost nothing, the small model wins on price alone. From about 1.3 cents per error the medium model is cheapest, and from about 9 cents the large one: its higher price per token is repaid by its fewer mistakes. Real products often sit far to the right of this chart, with errors that cost minutes of someone's time.`, chart: {"kind":"line","x":[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3],"points":false,"series":[{"name":"large model (92% correct)","y":[9,9.8,10.6,11.4,12.2,13,13.8,14.6,15.4,16.2,17,17.8,18.6,19.4,20.2,21,21.8,22.6,23.4,24.2,25,25.8,26.6,27.4,28.2,29,29.8,30.6,31.4,32.2,33]},{"name":"medium model (85% correct)","y":[2.4,3.9,5.4,6.9,8.4,9.9,11.4,12.9,14.4,15.9,17.4,18.9,20.4,21.9,23.4,24.9,26.4,27.9,29.4,30.9,32.4,33.9,35.4,36.9,38.4,39.9,41.4,42.9,44.4,45.9,47.4]},{"name":"small model (70% correct)","y":[0.4,3.41,6.41,9.4,12.41,15.41,18.41,21.41,24.41,27.41,30.41,33.41,36.41,39.41,42.41,45.41,48.41,51.41,54.41,57.41,60.41,63.41,66.41,69.41,72.41,75.41,78.41,81.41,84.41,87.41,90.41]}],"xLabel":"cost of one wrong answer (dollars)","yLabel":"total cost per 1,000 requests (dollars)","yMin":0} },
      { type: 'text', title: "Calling models: APIs and SDKs", md: `Every major provider offers an HTTP API and official SDKs for it. A request names a **model**, gives a **system prompt** and a list of **messages** with roles (user and assistant), and sets parameters such as the maximum number of output tokens, the sampling temperature and stop sequences (see [Decoding & Sampling](#/concept/decoding)). It can also define **tools** the model may call and a **schema** that the output must follow (see [Prompt Engineering & Structured Output](#/concept/prompting)). In production, a few more things matter:

- **Streaming.** Tokens arrive as server-sent events while they are generated, so the user sees text after a fraction of a second instead of waiting for the whole answer. Time to first token is the latency users notice most.
- **Errors and retries.** Expect rate limits (HTTP 429), overloads, timeouts and dropped connections. Retry with exponential backoff and jitter (next code), set timeouts, and make tool actions safe to repeat.
- **Batch APIs** process large offline jobs asynchronously, typically at about half the price.
- **Prompt caching** makes repeated prefixes much cheaper (see [LLM Inference & Serving](#/concept/llm-serving)).
- **OpenAI-compatible endpoints.** Many providers and local servers (vLLM, Ollama, LM Studio) accept the request format of OpenAI's Chat Completions API, so switching often means changing a base URL and a model name. **Routers and gateways** such as OpenRouter or LiteLLM put one API in front of many providers, with fallbacks and unified billing, at the price of one more party handling your data.
- **Keys and spending.** Keep API keys on a server, never in a web page or mobile app; use separate keys per environment, rotate them, and set spending limits.` },
      { type: 'code', title: "Retries with exponential backoff and jitter", md: `An API rejects every request for 30 seconds. How many requests does one client send while it waits, and what happens when 1,000 clients that failed at the same moment retry?`, code: `import random
random.seed(1)
OUTAGE = 30.0                                    # seconds during which the API rejects every request

def schedule(strategy, retries=400, base=1.0, cap=30.0, jitter=False):
    """The times at which one client sends requests, starting with a failed call at t = 0."""
    t, times = 0.0, [0.0]
    for i in range(retries):
        wait = {"every 0.1 s": 0.1, "every 1 s": 1.0}.get(strategy) or min(cap, base * 2 ** i)
        if jitter:
            wait = random.uniform(0, wait)       # "full jitter": a random wait between 0 and the backoff value
        t += wait
        if t >= OUTAGE:                          # the outage is over: no more retries needed
            break
        times.append(t)
    return times

print("strategy                 requests one client sends during a 30 s outage")
for s in ("every 0.1 s", "every 1 s", "exponential backoff"):
    print(f"{s:24s} {len(schedule(s)):5d}")
print("exponential backoff sends 5 requests:", len(schedule("exponential backoff")) == 5)

def busiest_100ms(jitter, clients=1000):         # 1,000 clients that all failed at the same moment
    counts = {}
    for _ in range(clients):
        for t in schedule("exponential backoff", jitter=jitter)[1:]:
            counts[int(t * 10)] = counts.get(int(t * 10), 0) + 1
    return max(counts.values())
without, with_jitter = busiest_100ms(False), busiest_100ms(True)
print(f"1,000 clients retrying: busiest 100 ms without jitter {without}, with jitter {with_jitter}")
print("jitter cuts the peak at least threefold:", with_jitter * 3 <= without)`, after: `Retrying every 0.1 seconds sends 300 requests into a service that is already overloaded; exponential backoff (waits of 1, 2, 4, 8 and 16 seconds) sends 5. But backoff alone keeps clients in lockstep: 1,000 clients that failed together all retry at exactly the same instants and hit the service with 1,000 requests at once, again and again. Random jitter spreads them out, cutting the peak about sixfold here. Official SDKs implement this for you; check that it is enabled and how many retries it makes.`, predict: {"type":"mcq","q":"During a 30-second outage, how many requests does one client send if it waits 1 s, 2 s, 4 s, … between attempts (starting with a failed call at t = 0)?","options":["About 300","About 30","5","1"],"answer":2,"explain":"Requests go out at 0, 1, 3, 7 and 15 seconds; the next one would be at 31 s, after the outage. Retrying every 0.1 s would send 300.","expect":["exponential backoff sends 5 requests: True"]} },
      { type: 'check', question: {"type":"mcq","q":"Why add random jitter to exponential backoff?","options":["To make each retry faster","So that many clients that failed together do not all retry at the same moment","To hide errors from the user","To lower the token count"],"answer":1,"explain":"Without jitter, synchronized retries arrive in waves that can overload the service again; randomizing the waits spreads them out."} },
      { type: 'text', title: "Running open models yourself", md: `- **The Hugging Face Hub** hosts hundreds of thousands of models and datasets, each with a model card describing its licence, training data, intended use and evaluations. The \`transformers\` library downloads and runs them, and hosted inference services run them for you.
- **On a laptop:** Ollama and LM Studio download quantized models, chat with them, and expose a local OpenAI-compatible API, and llama.cpp runs GGUF-format quantized models on CPUs and GPUs. Small models (a few billion parameters) run comfortably on a recent laptop.
- **On servers:** vLLM, SGLang and Hugging Face's Text Generation Inference serve models to many users at high throughput, with continuous batching and paged KV caches (see [LLM Inference & Serving](#/concept/llm-serving)).
- **In the browser:** Transformers.js runs small models in a web page with WebGPU or WebAssembly: private and free to serve, but limited to small models.
- **Will it fit?** Memory is about the number of parameters times the bytes per parameter, plus the KV cache; the calculator in [Quantization](#/concept/quantization) does the arithmetic.

Self-hosting pays off when data must stay in-house, when volume is high and steady, when you need to fine-tune deeply, or when you need a model that never changes. Otherwise an API is usually cheaper once engineering time is counted.` },
      { type: 'text', title: "Frameworks and agent SDKs", md: `- **LangChain** (with **LangGraph** for agent workflows) provides building blocks: model wrappers, prompt templates, document loaders, retrievers, tools and graphs of steps.
- **LlamaIndex** focuses on connecting models to your data: loading, indexing and querying documents for RAG. **Haystack** builds search and RAG pipelines, and **RAGFlow** is a complete RAG engine with document parsing.
- **Agent SDKs** from model providers, such as the OpenAI Agents SDK, Anthropic's Claude Agent SDK and Google's Agent Development Kit, run the agent loop: tool calls, hand-offs between agents, sessions, guardrails and tracing.

Frameworks give you a fast start and many ready-made integrations. Their costs are abstraction (the prompts and control flow are hidden, which makes debugging harder), rapid version churn, and code that is tied to the framework. A common path is to prototype with whatever is fastest, keep the core logic in plain code you understand, and adopt framework pieces where they clearly save work: a connector, a document parser, a tracing integration.` },
      { type: 'text', title: "AI-assisted coding", md: `Coding assistants and agents, such as GitHub Copilot, Cursor, Claude Code, OpenAI Codex and Replit's agent, complete code, explain it, and carry out multi-step changes across a codebase. Used well, they speed up both prototyping and maintenance:

- **Give precise tasks with context:** the relevant files, constraints, examples of the style you want, and what "done" means.
- **Make tests the judge.** Ask for tests first, or write them yourself, and run them on every change.
- **Review every diff** as if a colleague had written it, and work in small steps under version control.
- **Never paste secrets** into prompts, and check suggested dependencies: models sometimes invent package names, which attackers can register.
- **Keep understanding the code.** Assistants amplify skill; they do not replace knowing what the code does and why.` },
      { type: 'check', question: {"type":"mcq","q":"When is self-hosting an open-weight model usually the better choice than an API?","options":["Always, because it is free","When data must stay in-house, volume is high and steady, or deep fine-tuning or a frozen version is needed","When you have little traffic and want to launch quickly","Never, because open models are always weaker"],"answer":1,"explain":"Self-hosting trades per-token fees for hardware and operations work, which pays off for privacy, steady high volume or full control."} }
    ],
    when: `- Adding LLM features to a product: start with an API model, measure it on your own evaluation set, then optimize.
- Data must stay on your own hardware, or the volume is high and steady: consider open-weight models that you host.
- Many integrations or document loaders are needed quickly: a framework can save time. For a few calls, plain SDK code is simpler.`,
    pitfalls: `- Choosing a model by its leaderboard rank instead of your own evaluation set.
- Comparing price per token instead of cost per successful outcome.
- Retrying failed calls immediately and all at once, which turns a brief overload into an outage; use exponential backoff with jitter.
- Shipping API keys inside a web page or mobile app; keep them on a server.
- Not pinning model versions, so behaviour changes when the provider updates an alias.`,
    takeaways: [
      'AI engineers build products around pretrained models; ML engineers train and operate models. The skills overlap, but the daily work differs.',
      'Pick models with your own evaluation set and the cost of an outcome, not the price per token: a pricier model can be cheaper when errors are expensive.',
      'Robust API code streams responses, validates structured output, retries with exponential backoff and jitter, and pins model versions.',
      'Open-weight models run on laptops (Ollama, LM Studio, llama.cpp) or servers (vLLM), with full control over data and fine-tuning.',
      'Frameworks speed up prototypes with connectors and ready-made patterns; direct SDK calls keep prompts and control flow visible.'
    ],
    quiz: [
      {"type":"mcq","q":"What is the main focus of an AI engineer, compared with an ML engineer?","options":["Training models from scratch","Building reliable products on top of pretrained models: prompts, retrieval, tools, evaluation and integration","Designing GPUs","Writing statistical reports"],"answer":1,"explain":"AI engineers mostly start from existing models; ML engineers train, deploy and operate models."},
      {"type":"tf","q":"The model with the lowest price per token is always the cheapest choice for a product.","answer":false,"explain":"What matters is the cost per successful outcome. A cheaper model that makes more mistakes can cost more once errors are paid for."},
      {"type":"tf","q":"Many providers and local servers accept OpenAI-compatible requests, so switching between them often means changing only the base URL and the model name.","answer":true,"explain":"The Chat Completions request format has become a common interface, which is also what routers such as OpenRouter build on."},
      {"type":"mcq","q":"Where should an application's LLM API key be kept?","options":["In the JavaScript of the web page","On a server that calls the API on the user's behalf","In the mobile app's code","In the prompt"],"answer":1,"explain":"Anything shipped to a browser or a phone can be extracted; keys belong on a server, with spending limits."},
      {"type":"mcq","q":"Which is a real cost of building on a large framework such as LangChain?","options":["It cannot call LLMs","Its abstractions can hide the prompts and control flow, and its APIs change quickly","It only works offline","It forbids RAG"],"answer":1,"explain":"Frameworks speed up the start, but debugging hidden prompts and keeping up with version changes cost time later."},
      {"type":"mcq","q":"What makes a coding assistant's change safe to accept?","options":["That it compiles","That it passes tests you trust and you have reviewed the diff","That it is long","That the assistant says it is correct"],"answer":1,"explain":"Tests and review are the judge; an assistant's confidence is not evidence."}
    ],
    problems: [
      {
        id: 'request-cost', title: 'Cost of an API Request', difficulty: 'easy',
        prompt: `LLM APIs charge separately for input and output tokens, quoted in dollars **per million tokens**. Implement \`request_cost(input_tokens, output_tokens, price_in, price_out)\` that returns the cost of one request in dollars.

\`\`\`text
request_cost(1500, 300, 3.0, 15.0)   ->  0.009      1,500 × 3 / 10⁶ + 300 × 15 / 10⁶
\`\`\``,
        starter: `
def request_cost(input_tokens, output_tokens, price_in, price_out):
    # TODO: prices are per million tokens
    pass
`,
        solution: `
def request_cost(input_tokens, output_tokens, price_in, price_out):
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
`,
        hints: ['Divide by 1,000,000 because prices are per million tokens.', 'Input and output tokens have different prices: add the two parts.'],
        explain: `Output tokens usually cost several times more than input tokens, and long conversations or retrieved documents make the input large. Multiply by requests per day to get a budget, and divide by the success rate to get the cost of a useful answer.`,
        tests: [
          { call: 'round(request_cost(1500, 300, 3.0, 15.0), 6)', expect: '0.009' },
          { name: 'input only', call: 'round(request_cost(1_000_000, 0, 0.15, 0.6), 6)', expect: '0.15' },
          { name: 'output only', call: 'round(request_cost(0, 2000, 3, 15), 6)', expect: '0.03' },
          { name: 'odd sizes', call: 'round(request_cost(12345, 678, 0.8, 4.0), 8)', hidden: true }
        ]
      },
      {
        id: 'backoff-schedule', title: 'Exponential Backoff', difficulty: 'easy',
        prompt: `When an API answers "429: too many requests" or "overloaded", clients wait and try again, doubling the wait each time up to a maximum. Implement \`backoff_delays(retries, base=1.0, cap=30.0)\` that returns the list of waits, in seconds, before each of \`retries\` retries: base, 2·base, 4·base, …, never more than \`cap\`.

\`\`\`text
backoff_delays(5)                      ->  [1.0, 2.0, 4.0, 8.0, 16.0]
backoff_delays(7, base=0.5, cap=10)    ->  [0.5, 1.0, 2.0, 4.0, 8.0, 10, 10]
\`\`\`

(Real clients also add random *jitter* to each wait, so that many clients do not retry at the same moment.)`,
        starter: `
def backoff_delays(retries, base=1.0, cap=30.0):
    # TODO
    pass
`,
        solution: `
def backoff_delays(retries, base=1.0, cap=30.0):
    return [min(cap, base * 2 ** i) for i in range(retries)]
`,
        hints: ['The i-th wait is base · 2^i.', 'Use min(cap, …) to limit it.'],
        explain: `Doubling the wait backs off quickly when a service is overloaded, and the cap keeps a single request from waiting forever. Without jitter, thousands of clients that failed together retry together, and the overload repeats in waves.`,
        tests: [
          { call: 'backoff_delays(5)', expect: '[1.0, 2.0, 4.0, 8.0, 16.0]' },
          { name: 'a cap', call: 'backoff_delays(7, base=0.5, cap=10)', expect: '[0.5, 1.0, 2.0, 4.0, 8.0, 10, 10]' },
          { name: 'no retries', call: 'backoff_delays(0)', expect: '[]' },
          { name: 'another base', call: 'backoff_delays(6, base=0.25, cap=3)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Chip Huyen, AI Engineering (book, 2025): resources and notes', url: 'https://github.com/chiphuyen/aie-book' },
      { title: 'Hugging Face, Hub documentation', url: 'https://huggingface.co/docs/hub' },
      { title: 'Ollama, run open models locally', url: 'https://ollama.com' }
    ]
  },

  // =====================================================================================
  {
    id: 'llm-observability', branch: 'apps', title: 'LLMs in Production: Observability, Monitoring and Safety', short: 'LLMs in Production', level: 2, minutes: 45, prereqs: ['llm-evaluation', 'agents'],
    tags: ['observability', 'tracing', 'spans', 'logging', 'monitoring', 'latency', 'p95', 'p99', 'time to first token', 'cost monitoring', 'langsmith', 'langfuse', 'helicone', 'arize', 'phoenix', 'opentelemetry', 'deepeval', 'ragas', 'promptfoo', 'moderation', 'guardrails', 'user id', 'rate limiting', 'canary', 'regression testing'],
    summary: 'Running LLM applications reliably: tracing every request as a tree of spans; the metrics that matter (latency percentiles, time to first token, tokens and cost, errors, quality signals); continuous evaluation and regression testing; safety practices for production (moderation, end-user IDs, constrained inputs and outputs, rate limits, red teaming); and the observability and evaluation tools teams use.',
    why: `An LLM feature can fail without any error: the reply is fluent, fast and wrong. Requests branch into retrieval, several model calls and tools; costs differ a hundredfold between users; and every prompt or model change can quietly break something that used to work. Observability (seeing what each request actually did) and continuous evaluation are what turn a demo into a product that can be operated, debugged and improved.`,
    objectives: [
      'Trace LLM requests as spans with inputs, outputs, tokens, latency and cost, while protecting user privacy',
      'Monitor the metrics that matter, using percentiles rather than averages',
      'Set up continuous evaluation: deterministic checks, model-based judges, human review and regression tests',
      'Apply production safety practices: moderation, end-user IDs, constrained inputs and outputs, rate limits and red teaming'
    ],
    sections: [
      { type: 'text', title: "Why LLM applications need observability", md: `Traditional software fails loudly: an exception, a timeout, a 500 error. LLM applications often fail quietly:

- **A wrong answer looks like a right one.** It is fluent, fast and returns status 200.
- **One request is many steps:** retrieval, several model calls, tool calls, retries, sometimes an agent loop of dozens of steps.
- **Costs vary enormously:** one user's long conversation or runaway agent can cost a hundred times an ordinary request.
- **Behaviour changes without code changes:** a new model version, an edited prompt, new documents in the index or a shift in what users ask.

**Observability** means recording enough about every request that you can answer, afterwards, what happened and why: which documents were retrieved, what the model was actually sent, what it answered, which tools it called, how long each step took and what it cost.` },
      { type: 'text', title: "Traces and spans", md: `The standard way to record a request is a **trace**: a tree of **spans**, one span per step, each with a start, a duration and attributes. For LLM applications the useful attributes are:

- the **inputs and outputs** of each step (the prompt sent, the response, the retrieved chunks, the tool arguments and results);
- **versions**: model, prompt template, index and application;
- **tokens** in and out, and the **cost** they imply;
- **latency**, including the time to the first token for streamed responses;
- **errors, retries and refusals**;
- **feedback** linked to the trace: thumbs up or down, a correction, an escalation to a person.

Traces contain user data, so treat them as sensitive: redact personal information where possible, refer to users by **pseudonymous ids** rather than names or emails, restrict access, and set retention limits. OpenTelemetry, the common open standard for traces and metrics, has semantic conventions for generative-AI spans, so different tools can read the same data.` },
      { type: 'code', title: "A tiny tracer", md: `A minimal tracer built from a context manager, recording a simulated RAG request (the clock is simulated, so the example gives the same numbers every time):`, code: `import contextlib

clock = [0.0]                                        # a simulated clock in milliseconds, so the example is repeatable
def work(ms):
    clock[0] += ms                                   # pretend to do something that takes \`ms\` milliseconds

TRACE, stack = [], []
@contextlib.contextmanager
def span(name, kind, **attrs):
    record = {"name": name, "kind": kind, "depth": len(stack), "start": clock[0], **attrs}
    stack.append(record)
    try:
        yield record
    finally:
        stack.pop()
        record["ms"] = clock[0] - record["start"]
        TRACE.append(record)

PRICE_IN, PRICE_OUT = 3.0, 15.0                      # hypothetical dollars per million tokens

with span("handle request", "request", user="u_7f3a"):              # a pseudonymous user id, never the email
    with span("embed question", "retrieval"):
        work(35)
    with span("vector search", "retrieval", hits=4):
        work(22)
    with span("generate answer", "llm", model="model-large-2026-01", input_tokens=2_400, output_tokens=310):
        work(380 + 310 * 9)                          # 380 ms to the first token, then about 9 ms per token
    with span("check citations", "tool"):
        work(4)

total = max(s["ms"] for s in TRACE)
cost = 0.0
for s in sorted(TRACE, key=lambda s: (s["start"], s["depth"])):
    extra = ""
    if s["kind"] == "llm":
        c = (s["input_tokens"] * PRICE_IN + s["output_tokens"] * PRICE_OUT) / 1e6
        cost += c
        extra = f"   {s['input_tokens']:,} in / {s['output_tokens']} out tokens   \${c:.5f}"
    print(f"{'  ' * s['depth']}{s['name']:<{24 - 2 * s['depth']}} {s['ms']:7,.0f} ms{extra}")
llm_share = sum(s["ms"] for s in TRACE if s["kind"] == "llm") / total
print(f"the model call takes {llm_share:.0%} of the request; cost of the request \${cost:.5f}")
print("generation takes more than 80% of the time:", llm_share > 0.8)`, after: `The trace shows at once where the time goes: retrieval and the citation check take about 60 milliseconds together, while the model call takes over 3 seconds, mostly spent producing its 310 output tokens. To make this request feel faster, stream the answer (the first token arrives after 380 ms), ask for shorter answers or use a faster model, not a faster vector database. Observability tools record exactly this tree for every request and let you search and aggregate them.`, predict: {"type":"mcq","q":"In this RAG request, which step takes most of the time?","options":["Embedding the question","The vector search","Generating the answer","Checking the citations"],"answer":2,"explain":"Generation takes about 3.17 of the 3.23 seconds (98%): 380 ms until the first token and about 9 ms for each of its 310 output tokens. Retrieval takes under 60 ms.","expect":["generation takes more than 80% of the time: True"]} },
      { type: 'text', title: "Metrics that matter", md: `Aggregated over many traces, the useful production metrics are:

- **Latency percentiles** (p50, p95, p99) and the **time to first token**, per feature and per model. Averages hide the slow tail (next code).
- **Tokens and cost** per request, per user and per feature, plus the cache hit rate when prompt caching is used, and alerts for unusual spikes (an agent stuck in a loop, an abusive user).
- **Errors and degradations:** API errors, timeouts, retries, refusals, malformed structured output, fallback to a backup model.
- **Quality signals:** explicit feedback, regenerations, abandoned conversations, escalations to people, and scores from evaluations run on samples of real traffic.
- **Drift:** changes in the topics, languages or lengths of what users ask, which can make your evaluation set stale.` },
      { type: 'code', title: "Percentiles, not averages", md: `LLM latencies have a long tail: most requests are quick, a few are very slow (long answers, retries, overloaded providers). Ten thousand simulated requests:`, code: `import numpy as np
rng = np.random.default_rng(0)
latency = rng.lognormal(mean=np.log(1.6), sigma=0.6, size=10_000)     # seconds; long-tailed, like real LLM latencies
p50, p95, p99 = np.percentile(latency, [50, 95, 99])
print(f"mean {latency.mean():.2f} s   p50 {p50:.2f} s   p95 {p95:.2f} s   p99 {p99:.2f} s")
slow = int(np.sum(latency > 5))
print(f"requests slower than 5 s: {slow / len(latency):.1%}, that is {slow} of 10,000 requests")
print("p99 is more than three times the mean:", bool(p99 > 3 * latency.mean()))`, after: `The average of under 2 seconds looks fine, but one request in twenty takes more than 4 seconds and one in a hundred more than 6. At a million requests a day, that is ten thousand people a day waiting over 6 seconds. Set service-level objectives on percentiles ("95% of answers start streaming within 1 second"), and investigate the slowest traces, not the average.`, predict: {"type":"mcq","q":"The latencies average about 1.9 seconds. Roughly where is the 99th percentile?","options":["About 2 seconds","About 3 seconds","About 6 seconds or more","About 20 seconds"],"answer":2,"explain":"With a long-tailed distribution the slowest 1% sits far above the mean: about 6.4 seconds here, more than three times the average.","expect":["p99 is more than three times the mean: True"]} },
      { type: 'chart', title: "How the latencies are distributed", md: `The 10,000 simulated latencies from the code above:`, after: `Most requests cluster around 1 to 2 seconds, but the distribution has a long right tail. The median sits left of the mean, and p95 and p99 lie far out in the tail, where the users who complain are. A dashboard that shows only the average would look the same whether that tail were short or long.`, chart: {"kind":"line","x":[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9],"points":false,"series":[{"name":"share of requests in each 0.2-second bin","y":[0.02,1.09,3.85,7.21,9.2,10.22,9.76,8.91,6.96,6.67,5.57,5.11,4.25,3.34,2.98,2.37,2.06,1.66,1.35,1.08,0.95,0.78,0.62,0.56,0.53,0.48,0.35,0.25,0.24,0.17,0.23,0.17,0.14,0.17,0.11,0.08,0.07,0.06,0.02,0.09]}],"xLabel":"latency (seconds)","yLabel":"requests (%)","yMin":0,"vlines":[{"x":1.59,"label":"p50"},{"x":4.28,"label":"p95"},{"x":6.4,"label":"p99"}]} },
      { type: 'text', title: "Continuous evaluation and regression testing", md: `Evaluation does not end at launch (see [Evaluating LLM Applications](#/concept/llm-evaluation)). In production it runs in layers:

- **Deterministic checks on every request**, cheap and exact: valid JSON or schema, required fields present, length limits, no leaked system prompt, links and citations that resolve.
- **Model-based evaluation on samples:** an LLM judge scores faithfulness, helpfulness or tone for a sample of real traffic, calibrated regularly against human labels.
- **Human review** of a small random sample and of every flagged case, which also feeds new test cases.
- **Regression suites in CI:** every change of prompt, model, retrieval settings or tools runs the offline evaluation set, and a drop beyond the noise blocks the change. Failures found in production become new test cases.
- **Gradual rollouts:** release changes to a small share of traffic first (a canary), compare its metrics with the rest, and roll back quickly if they worsen.` },
      { type: 'check', question: {"type":"mcq","q":"A team edits a system prompt to fix one complaint. What should happen before the change reaches all users?","options":["Nothing; prompt edits are not code changes","Run the regression evaluation suite and roll the change out gradually while watching the metrics","Ask the model whether the new prompt is better","Delete the old traces"],"answer":1,"explain":"Prompt edits change behaviour as much as code does, and a fix for one case often breaks others. The regression suite and a canary catch that."} },
      { type: 'text', title: "Safety in production", md: `Safety for a deployed LLM product is a set of overlapping layers, none of them sufficient alone:

- **Know your users and use cases.** Decide what the product is for and what it must refuse, and design the defences around those risks.
- **Moderation on inputs and outputs.** Classifiers flag harmful content (violence, self-harm, sexual content involving minors, harassment) before a prompt reaches the model and before a response reaches the user. Providers offer moderation endpoints, and open models such as Llama Guard can be self-hosted.
- **End-user ids.** Attach a stable, pseudonymous id for the end user to each request, which most providers support. It lets you, and the provider's abuse systems, trace misuse to an account and block it without blocking your whole application.
- **Constrain inputs and outputs.** Limit input length and accepted formats, use structured output and allow-lists, validate model output before acting on it, and never execute model-written code or queries without sandboxing.
- **Rate limits and budgets** per user and per feature against abuse and runaway costs.
- **Prompt injection defences** for anything that reads untrusted content (see [Prompt Engineering & Structured Output](#/concept/prompting) and [Advanced Agents](#/concept/advanced-agents)).
- **Adversarial testing before launch and after major changes:** red-team the system with jailbreaks, injections and abuse scenarios. The OWASP Top 10 for LLM applications is a useful checklist.
- **An incident process:** who is alerted, how a feature is switched off quickly, and how affected users are informed.` },
      { type: 'text', title: "Tools of the trade", md: `- **Tracing and observability:** LangSmith (from the LangChain team), Langfuse (open source), Helicone (a proxy in front of the provider's API that logs every call), Arize Phoenix (open source) and Weights & Biases Weave, many of them able to read OpenTelemetry data, alongside general monitoring tools such as Datadog or Grafana.
- **Evaluation:** DeepEval and promptfoo for test suites in CI, RAGAS for metrics specific to RAG (faithfulness, context precision and recall), and the evaluation features of the tracing tools.
- **Guardrails:** moderation endpoints, Llama Guard, and libraries that validate inputs and outputs against rules or schemas.

Which tool hardly matters at first. What matters is to trace from day one, keep evaluation sets in version control next to the prompts, and make every production failure a new test case.` },
      { type: 'check', question: {"type":"tf","q":"Logging every prompt and response in full, with users' email addresses and no retention limit, is good observability practice.","answer":false,"explain":"Traces are sensitive data: use pseudonymous user ids, redact personal information where possible, restrict access and set retention limits."} }
    ],
    when: `- Before launch: tracing, a regression suite and dashboards for cost and latency should exist on day one.
- When debugging a bad answer: open its trace and see which retrieval, prompt or tool call went wrong.
- Before every change of prompt, model or retrieval settings: run the regression suite and compare.`,
    pitfalls: `- Reporting average latency: users feel the slow tail (p95, p99).
- Logging prompts and outputs that contain personal data, with no redaction or retention limit.
- Changing a prompt in production without running the regression suite.
- Relying on a model-based judge that was never checked against human labels.
- Treating moderation as optional for user-facing generation.`,
    takeaways: [
      'Trace every request as a tree of spans (retrieval, model calls, tools) with tokens, latency, cost and versions.',
      'Watch percentiles: a 2-second mean can hide a p99 several times longer.',
      'Evaluate continuously: cheap deterministic checks on everything, model-based judges on samples, humans for calibration, and a regression suite on every change.',
      'Production safety is layered: moderation, end-user IDs, constrained inputs and outputs, rate limits, red teaming and an incident process.',
      'Tools such as LangSmith, Langfuse, Helicone and Arize Phoenix (often built on OpenTelemetry) collect traces; DeepEval, RAGAS and promptfoo run evaluations.'
    ],
    quiz: [
      {"type":"mcq","q":"What is a span in a trace?","options":["A whole user session","One step of a request (a retrieval, a model call, a tool call) with its duration and attributes","The context window","A type of model"],"answer":1,"explain":"A trace is a tree of spans; each span records one step, so you can see where time, tokens and errors come from."},
      {"type":"tf","q":"For a feature with long-tailed latency, the average is the best number to put in a service-level objective.","answer":false,"explain":"The average hides the slow tail that users feel. Objectives are set on percentiles such as p95 or p99."},
      {"type":"mcq","q":"Why attach a pseudonymous end-user id to each API request?","options":["To make responses faster","So that misuse can be traced to one account and blocked without blocking the whole application","To personalize the model's weights","It is required for streaming"],"answer":1,"explain":"With per-user ids, you and the provider can act on abuse precisely instead of suspending your application's key."},
      {"type":"mcq","q":"Which check belongs in the cheap, deterministic layer that runs on every request?","options":["Asking an LLM judge whether the answer is helpful","Checking that the output is valid JSON matching the schema","Asking a human reviewer","Retraining the model"],"answer":1,"explain":"Schema, format, length and link checks are exact and nearly free, so they can run on every request; judges and humans review samples."},
      {"type":"mcq","q":"A trace shows retrieval taking 60 ms and generation 3,100 ms. What would most improve perceived latency?","options":["A faster vector database","Streaming the answer and asking for shorter outputs","More retrieved chunks","A larger model"],"answer":1,"explain":"Generation dominates. Streaming shows the first tokens quickly, and shorter outputs cut the total time."},
      {"type":"fill","q":"With the nearest-rank method, what is the p90 of the latencies [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] seconds?","answer":["9"],"explain":"The rank is ⌈0.9 × 10⌉ = 9, and the 9th value in sorted order is 9."}
    ],
    problems: [
      {
        id: 'nearest-rank-percentile', title: 'Latency Percentiles', difficulty: 'easy',
        prompt: `Dashboards report latency **percentiles**: p95 is the latency that 95% of requests stay under. Implement \`percentile(values, q)\` with the **nearest-rank** method:

- sort the values;
- the rank is ⌈q / 100 × n⌉ (round up), but at least 1;
- return the value at that rank (counting from 1).

\`\`\`text
percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)   ->  5
percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95)   ->  10
\`\`\``,
        starter: `
import math

def percentile(values, q):
    # TODO: nearest-rank percentile
    pass
`,
        solution: `
import math

def percentile(values, q):
    s = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(s)))
    return s[rank - 1]
`,
        hints: ['math.ceil rounds the rank up.', 'Ranks count from 1, list indices from 0: return s[rank − 1].'],
        explain: `The nearest-rank percentile always returns a value that really occurred, which suits latency dashboards. With few requests the high percentiles are noisy: p99 of 50 requests is simply the slowest one.`,
        tests: [
          { call: 'percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)', expect: '5' },
          { name: 'p95', call: 'percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95)', expect: '10' },
          { name: 'unsorted input', call: 'percentile([9, 1, 8, 2, 7, 3], 50)', expect: '3' },
          { name: 'a single value', call: 'percentile([5], 99)', expect: '5' },
          { name: 'random latencies vs reference', setup: 'import random\nrandom.seed(7)\nlat = [round(random.lognormvariate(0, 0.6), 3) for _ in range(501)]', call: '[percentile(lat, q) for q in (50, 90, 95, 99)]', hidden: true }
        ]
      },
      {
        id: 'trace-summary', title: 'Summarize a Trace', difficulty: 'medium',
        prompt: `A trace is a list of spans, one per step of a request. Each span is a dict with a \`name\`, a \`kind\` (\`"llm"\`, \`"retrieval"\` or \`"tool"\`), its duration \`ms\`, and optionally \`input_tokens\` and \`output_tokens\`. Implement \`summarize(spans)\` that returns a dict:

- \`"total_ms"\`: the sum of all durations;
- \`"llm_calls"\`: how many spans have kind \`"llm"\`;
- \`"tokens"\`: the sum of input and output tokens over all spans (missing counts are 0);
- \`"slowest"\`: the name of the slowest span (the first one on a tie).`,
        starter: `
def summarize(spans):
    # TODO
    pass
`,
        solution: `
def summarize(spans):
    slowest = None
    for s in spans:
        if slowest is None or s["ms"] > slowest["ms"]:
            slowest = s
    return {
        "total_ms": sum(s["ms"] for s in spans),
        "llm_calls": sum(1 for s in spans if s["kind"] == "llm"),
        "tokens": sum(s.get("input_tokens", 0) + s.get("output_tokens", 0) for s in spans),
        "slowest": slowest["name"] if slowest else None,
    }
`,
        hints: ['s.get("input_tokens", 0) handles spans without token counts.', 'Keep the slowest span seen so far, and replace it only when a span is strictly slower.'],
        explain: `These are the first numbers to look at when a request is slow or expensive: how many model calls it made, how many tokens they used, and which step took the time. Observability tools compute them for every trace and aggregate them across users and features.`,
        tests: [
          { name: 'a RAG request', setup: 'spans = [{"name": "embed query", "kind": "retrieval", "ms": 40}, {"name": "vector search", "kind": "retrieval", "ms": 25}, {"name": "generate", "kind": "llm", "ms": 1800, "input_tokens": 2100, "output_tokens": 250}]', call: 'summarize(spans)', expect: "{'total_ms': 1865, 'llm_calls': 1, 'tokens': 2350, 'slowest': 'generate'}" },
          { name: 'an agent with tools', setup: 'spans = [{"name": "plan", "kind": "llm", "ms": 900, "input_tokens": 1200, "output_tokens": 80}, {"name": "search", "kind": "tool", "ms": 900}, {"name": "answer", "kind": "llm", "ms": 700, "input_tokens": 2500, "output_tokens": 300}]', call: 'summarize(spans)', expect: "{'total_ms': 2500, 'llm_calls': 2, 'tokens': 4080, 'slowest': 'plan'}" },
          { name: 'an empty trace', call: 'summarize([])', expect: "{'total_ms': 0, 'llm_calls': 0, 'tokens': 0, 'slowest': None}", hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'OpenTelemetry, Semantic conventions for generative AI systems', url: 'https://opentelemetry.io/docs/specs/semconv/gen-ai/' },
      { title: 'Langfuse, open-source LLM observability (documentation)', url: 'https://langfuse.com/docs' },
      { title: 'OWASP, Top 10 for Large Language Model Applications', url: 'https://owasp.org/www-project-top-10-for-large-language-model-applications/' }
    ]
  },

  // =====================================================================================
  {
    id: 'mcp', branch: 'apps', title: 'The Model Context Protocol (MCP)', short: 'MCP', level: 2, minutes: 40, prereqs: ['agents'],
    tags: ['mcp', 'model context protocol', 'mcp server', 'mcp client', 'mcp host', 'json-rpc', 'tools', 'resources', 'prompts', 'sampling', 'elicitation', 'stdio', 'streamable http', 'transport', 'oauth', 'tool poisoning', 'fastmcp'],
    summary: 'An open standard for connecting LLM applications to tools and data: hosts, clients and servers; the data layer (JSON-RPC 2.0 messages, the initialize handshake, tools, resources and prompts, and client features such as sampling and elicitation); the transport layer (stdio for local servers, Streamable HTTP for remote ones); building and connecting servers; and the security risks of letting third-party tools into a model\'s context.',
    why: `Before MCP, every AI application had to integrate every tool separately: N applications and M tools meant N × M custom integrations. MCP defines one protocol, so a server written once (for a database, a code repository, a calendar) works in any compatible assistant, IDE or agent framework. It has become a common way to give models access to the outside world, so understanding it, and its security implications, is part of an AI engineer's toolkit.`,
    objectives: [
      'Explain the roles of MCP hosts, clients and servers, and when to use local or remote servers',
      'Read and write the JSON-RPC messages of an MCP session: initialize, tools/list and tools/call',
      'Distinguish the server primitives (tools, resources, prompts) and the transports (stdio, Streamable HTTP)',
      'Assess the security risks of MCP servers and apply mitigations'
    ],
    sections: [
      { type: 'text', title: "Why a protocol?", md: `An LLM application becomes useful when it can reach the outside world: read files, query a database, search issues, create calendar events. Before a shared standard, every application wrote its own integration for every service, and every service had to integrate with every application: N applications and M tools meant up to N × M pieces of glue code.

The **Model Context Protocol** (MCP), released as an open standard by Anthropic in November 2024 and since adopted by many assistants, IDEs, agent frameworks and model providers, replaces that with one protocol. A service is wrapped once as an **MCP server**; any application that speaks MCP can then use it. It plays the role for AI applications that USB plays for devices, or the Language Server Protocol plays for code editors: write the connector once, plug it in anywhere.

MCP builds on [tool calling](#/concept/agents): the model still decides to call a tool and the application still executes it. What MCP standardizes is how the application *discovers* tools, data and prompts from external programs, and how it talks to them.` },
      { type: 'text', title: "Hosts, clients and servers", md: `| Role | What it is | Examples |
| --- | --- | --- |
| **Host** | The AI application the user works with. It owns the model, the conversation and the user's permissions. | a chat assistant, an IDE, a coding agent, your own agent |
| **Client** | A connection manager inside the host. A host creates one client per server it connects to. | part of the host's code, usually from an MCP SDK |
| **Server** | A program that exposes capabilities to clients. | a server for a file system, a database, GitHub, a calendar, a company's internal API |

Servers run in two ways:

- **Local servers** run on the user's machine as a subprocess of the host: fast, with access to local files and tools, but running with the user's own permissions.
- **Remote servers** run as web services, typically operated by the company whose product they expose, and are reached over HTTP with authorization.

Because each server is separate, a host can combine many of them, and each one sees only its own part of the work.` },
      { type: 'text', title: "The data layer: JSON-RPC messages and primitives", md: `All MCP messages are **JSON-RPC 2.0**: small JSON objects of three kinds.

- **Requests** carry an \`id\`, a \`method\` and \`params\`, and expect a response.
- **Responses** carry the same \`id\` and either a \`result\` or an \`error\` (with a numeric \`code\` and a \`message\`).
- **Notifications** have no \`id\` and get no response; they announce events such as "the list of tools changed".

A session starts with a **handshake**: the client sends \`initialize\` with the protocol version it supports, its capabilities and its name; the server answers with its own version, capabilities and name; the client then sends the notification \`notifications/initialized\`. After that, either side can send the requests its partner declared it supports.

What a server can offer, its **primitives**:

| Primitive | Controlled by | Methods | Example |
| --- | --- | --- | --- |
| **Tools** | the model, which decides when to call them | \`tools/list\`, \`tools/call\` | create an issue, run a query |
| **Resources** | the application, which decides what to include in the context | \`resources/list\`, \`resources/read\` | a file, a database schema, a document |
| **Prompts** | the user, who picks them, for example from a menu | \`prompts/list\`, \`prompts/get\` | "review this pull request" |

Clients can offer features to servers too: **sampling** (the server asks the host's model to generate text, so a server can use AI without its own API key), **elicitation** (the server asks the user for missing information), and **roots** (the directories a server may work in).` },
      { type: 'code', title: "An MCP session, message by message", md: `A toy server and client exchanging real MCP message shapes, without any network: the handshake, discovering tools, a successful call, a tool that fails, and a request for a tool that does not exist.`, code: `import json

class TinyServer:
    """A toy MCP server: it answers JSON-RPC requests about its one tool."""
    def __init__(self):
        self.tools = {"get_forecast": {
            "description": "Weather forecast for a city.",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
        self.forecasts = {"oslo": "rain, 9 °C", "lisbon": "sunny, 22 °C"}

    def handle(self, msg):
        if "id" not in msg:                                  # a notification: no reply
            return None
        reply = {"jsonrpc": "2.0", "id": msg["id"]}
        method, params = msg["method"], msg.get("params", {})
        if method == "initialize":
            reply["result"] = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                               "serverInfo": {"name": "weather", "version": "1.0"}}
        elif method == "tools/list":
            reply["result"] = {"tools": [dict(name=n, **t) for n, t in self.tools.items()]}
        elif method == "tools/call":
            if params["name"] not in self.tools:             # a problem with the request itself: a protocol error
                reply["error"] = {"code": -32602, "message": "Unknown tool: " + params["name"]}
            else:
                city = params["arguments"].get("city", "").lower()
                if city in self.forecasts:
                    reply["result"] = {"content": [{"type": "text", "text": self.forecasts[city]}], "isError": False}
                else:                                        # the tool ran but failed: report it inside the result
                    reply["result"] = {"content": [{"type": "text", "text": f"No forecast for '{city}'."}], "isError": True}
        else:
            reply["error"] = {"code": -32601, "message": "Method not found"}
        return reply

server, next_id = TinyServer(), 0
def send(method, params=None, notify=False):
    global next_id
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        next_id += 1
        msg["id"] = next_id
    print("client ->", json.dumps(msg))
    reply = server.handle(msg)
    if reply is not None:
        print("server <-", json.dumps(reply, ensure_ascii=False))
    return reply

send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "demo-host", "version": "0.1"}})
send("notifications/initialized", notify=True)
send("tools/list")
send("tools/call", {"name": "get_forecast", "arguments": {"city": "Oslo"}})
failed = send("tools/call", {"name": "get_forecast", "arguments": {"city": "Atlantis"}})
unknown = send("tools/call", {"name": "book_flight", "arguments": {}})
print("a failing tool is reported in the result:", failed["result"]["isError"])
print("an unknown tool is a protocol error with code:", unknown["error"]["code"])`, after: `The host learns what the server can do from \`tools/list\`, including each tool's JSON Schema, which it passes to the model as tool definitions. When the model asks for a forecast, the host sends \`tools/call\` and gives the model the content of the result. The two failures are deliberately different: a tool that runs and fails returns its error message *inside the result*, marked \`isError\`, so the model can read it and react (ask for another city), while a request the server cannot handle at all, such as an unknown tool, is a JSON-RPC protocol error for the host.`, predict: {"type":"mcq","q":"The model asks the forecast tool about a city the server has no data for. What does the server send back?","options":["A JSON-RPC error, as for a malformed request","A normal result whose text explains the failure, marked isError: true","Nothing at all","An empty forecast with no indication of failure"],"answer":1,"explain":"Tool execution errors are reported inside the result so the model can see them and try something else; JSON-RPC errors are reserved for problems with the request itself, such as an unknown tool.","expect":["a failing tool is reported in the result: True","an unknown tool is a protocol error with code: -32602"]} },
      { type: 'check', question: {"type":"mcq","q":"Which MCP primitive is controlled by the application rather than by the model, and supplies context such as files or database schemas?","options":["Tools","Resources","Prompts","Sampling"],"answer":1,"explain":"Resources are data the application chooses to include in the context; tools are called when the model decides, and prompts are picked by the user."} },
      { type: 'text', title: "Transports: stdio and Streamable HTTP", md: `The **transport layer** carries the same JSON-RPC messages in two standard ways:

- **stdio** for local servers: the host starts the server as a subprocess and exchanges newline-delimited JSON messages over its standard input and output. The server writes logs to standard error, never to standard output, which would corrupt the message stream.
- **Streamable HTTP** for remote servers: the client POSTs each message to a single endpoint, and the server replies either with plain JSON or with a stream of server-sent events, when it needs to send progress updates or several messages. A session id in an HTTP header ties the requests of one session together. (It replaced an earlier transport based on separate HTTP and SSE endpoints.)

Remote servers need **authorization**. MCP specifies an OAuth 2.1-based flow, so a user can grant a host access to, say, their calendar with scoped tokens, without giving the host their password and without the server passing the user's tokens on to other services.` },
      { type: 'code', title: "A real server with the Python SDK", run: false, md: `The official SDKs (Python, TypeScript and several other languages) take care of the protocol. A notes server with one tool, one resource and one prompt, to run on your own computer:`, code: `# pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes")
NOTES = {}

@mcp.tool()
def add_note(title: str, text: str) -> str:
    """Save a note under a title."""
    NOTES[title] = text
    return f"Saved '{title}'."

@mcp.resource("notes://{title}")
def read_note(title: str) -> str:
    """The text of one saved note."""
    return NOTES.get(title, "No such note.")

@mcp.prompt()
def summarize_notes() -> str:
    """Ask the model to summarize every note."""
    return "Summarize these notes:\\n" + "\\n".join(f"- {t}: {x}" for t, x in NOTES.items())

if __name__ == "__main__":
    mcp.run()      # stdio by default; mcp.run(transport="streamable-http") serves it over HTTP`, after: `The SDK turns the type hints and docstrings into each tool's name, description and input schema, answers \`initialize\`, \`tools/list\` and \`tools/call\`, and validates arguments. To connect it, tell the host how to start it. Many hosts use a configuration file along these lines, with the server's command and arguments:

\`\`\`json
{
  "mcpServers": {
    "notes": { "command": "python", "args": ["/path/to/notes_server.py"] }
  }
}
\`\`\`

The host then starts the server, runs the handshake and offers its tools to the model. SDKs also include an MCP **client**, for building your own host or agent that connects to existing servers.` },
      { type: 'code', title: "How much context do tool definitions use?", md: `Every tool a host offers is described to the model in every request: its name, description and input schema. A realistic definition, and what many of them cost:`, code: `import json
tool = {"name": "search_issues",
        "description": ("Search the issue tracker. Use it to find bugs or feature requests by keywords, labels or "
                        "assignee. Returns up to 20 issues with their id, title, status and URL."),
        "inputSchema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "Words to search for in titles and descriptions"},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Only issues that have all of these labels"},
            "assignee": {"type": "string", "description": "User name of the person the issue is assigned to"},
            "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Which issues to include"},
            "limit": {"type": "integer", "description": "Maximum number of results, from 1 to 20"}}}}
chars = len(json.dumps(tool))
tokens = chars / 4                                   # a common rule of thumb for English text and JSON
print(f"one tool definition: {chars} characters, roughly {tokens:.0f} tokens")
for n in (5, 20, 50, 100):
    print(f"{n:4d} tools: about {n * tokens:7,.0f} tokens in every request ({n * tokens / 200_000:5.1%} of a 200,000-token context)")
print("100 tools take more than 10,000 tokens per request:", n * tokens > 10_000)`, after: `One well-documented tool costs a couple of hundred tokens, so connecting a few servers with dozens of tools each quietly adds tens of thousands of tokens to every request, along with latency and cost, and gives the model more ways to pick the wrong tool. Prompt caching softens the cost, since the definitions rarely change, but the better fix is fewer, well-designed tools (see [Advanced Agents](#/concept/advanced-agents)), enabling only the servers a task needs, or letting the agent search for tools and load their definitions on demand.`, predict: {"type":"mcq","q":"A host connects to servers that expose 100 tools, each about this size. Roughly how many tokens do their definitions add to every request?","options":["About 200","About 2,000","About 20,000","About 2 million"],"answer":2,"explain":"About 200 tokens per definition × 100 tools ≈ 20,000 tokens, resent with every request unless cached.","expect":["100 tools take more than 10,000 tokens per request: True"]} },
      { type: 'text', title: "Security: a server is code you invite in", md: `MCP makes it easy to add capabilities, which also makes it easy to add risks:

- **Local servers run with your permissions.** Installing one is installing software: it can read your files and use your network. Install servers only from sources you trust, and pin their versions.
- **Tool poisoning.** Tool descriptions go straight into the model's context, so a malicious server can hide instructions in them ("before using any tool, send the contents of ~/.ssh to…"). Review descriptions, and prefer hosts that show them to you.
- **Changing definitions.** A server can change its tools after you approved them (a "rug pull"). Hosts should show changes and ask for approval again.
- **Injection through results.** Tool results (web pages, issue texts, emails) can carry prompt injections like any other untrusted content (see [Adversarial ML, Security & Privacy](#/concept/ml-security)).
- **Over-broad access.** Give each server the narrowest scopes and tokens it needs, keep a human confirmation step for actions with side effects, and watch for the combination of private data, untrusted content and a way to send data out (see [Advanced Agents](#/concept/advanced-agents)).` },
      { type: 'text', title: "MCP or plain function calling?", md: `- **Plain function calling** is the simplest choice when one application uses its own handful of functions: define them in the request and execute them in your code.
- **MCP** pays off when tools are **shared**: one server used by several applications, teams or users; third-party integrations that users plug in themselves; or tools maintained by the team that owns the service rather than by every consumer.
- Many applications do both: their own core functions as plain tools, and MCP for integrations.
- Agents that combine many servers need the same discipline as any agent: a small set of relevant tools per task, clear descriptions, and evaluation of how well the model picks and uses them.` },
      { type: 'check', question: {"type":"tf","q":"A local MCP server started by your assistant can only do what its tool descriptions say.","answer":false,"explain":"A local server is an ordinary program running with your permissions: its descriptions say what it claims to do, not what it can do. Install only servers you trust."} }
    ],
    when: `- Tools or data sources are shared across several AI applications, teams or users.
- You want users to plug third-party integrations into your assistant or agent.
- For a single application's own few functions, plain function calling is simpler.`,
    pitfalls: `- Installing servers from unknown sources: a local server runs arbitrary code with your permissions.
- Trusting tool descriptions and results: both can carry prompt injections ("tool poisoning").
- Exposing dozens of tools at once: every definition costs context tokens and makes the choice of tool harder.
- Passing users' tokens straight through to other services instead of using properly scoped credentials.`,
    takeaways: [
      'MCP standardizes how applications (hosts) connect, through clients, to servers that expose tools, resources and prompts.',
      'Messages are JSON-RPC 2.0: an initialize handshake, then requests such as tools/list and tools/call, plus notifications.',
      'Local servers talk over stdio; remote servers use Streamable HTTP with OAuth-based authorization.',
      'Tool failures come back inside the result (isError) so the model can react; protocol problems use JSON-RPC error codes.',
      'Treat every server as code you invite in: vet it, pin its version, scope its permissions and confirm side effects.'
    ],
    quiz: [
      {"type":"mcq","q":"What problem does MCP mainly solve?","options":["It makes models more accurate","It replaces N × M custom integrations between applications and tools with one shared protocol","It trains models to use tools","It encrypts model outputs"],"answer":1,"explain":"A tool wrapped once as an MCP server works in every MCP-compatible host."},
      {"type":"mcq","q":"In MCP, what is a host?","options":["The machine a server runs on","The AI application the user interacts with, which creates one client per server connection","A remote server","A JSON-RPC message"],"answer":1,"explain":"Hosts such as assistants, IDEs and agents own the model and the conversation; their clients connect to servers."},
      {"type":"tf","q":"MCP messages are JSON-RPC 2.0 requests, responses and notifications.","answer":true,"explain":"Requests carry an id and expect a response; notifications have no id."},
      {"type":"mcq","q":"Which transport does a local MCP server usually use?","options":["Email","stdio: JSON messages over the standard input and output of a subprocess","WebRTC","FTP"],"answer":1,"explain":"Local servers run as subprocesses of the host and talk over stdio; remote servers use Streamable HTTP."},
      {"type":"mcq","q":"What is tool poisoning?","options":["A tool that returns wrong numbers","Hidden instructions placed in a tool's description or results that try to steer the model","Too many tools in one server","A server that crashes"],"answer":1,"explain":"Tool descriptions and results enter the model's context, so malicious text there acts like a prompt injection."},
      {"type":"mcq","q":"When is plain function calling a better choice than MCP?","options":["When tools are shared across many applications","When one application uses a few of its own functions","When users want to add third-party integrations","Never"],"answer":1,"explain":"For a single application's own functions, defining tools directly in the request is simpler; MCP pays off when tools are shared."}
    ],
    problems: [
      {
        id: 'mcp-dispatch', title: 'A Minimal MCP Request Handler', difficulty: 'medium',
        prompt: `Implement the core of an MCP server: \`handle(message, tools)\` takes a JSON-RPC request (a dict) and returns the response dict. \`tools\` maps each tool name to a pair \`(description, function)\`. Every response has \`"jsonrpc": "2.0"\` and the request's \`"id"\`.

- \`"tools/list"\` → \`"result": {"tools": [{"name": …, "description": …}, …]}\`, in the order of \`tools\`.
- \`"tools/call"\` with \`"params": {"name": …, "arguments": {…}}\`:
  - unknown tool → \`"error": {"code": -32602, "message": "Unknown tool: <name>"}\`;
  - otherwise call the function with the arguments as keyword arguments. On success: \`"result": {"content": [{"type": "text", "text": str(value)}], "isError": False}\`. If the function raises: the same shape with the exception's message as text and \`"isError": True\`.
- Any other method → \`"error": {"code": -32601, "message": "Method not found"}\`.`,
        starter: `
def handle(message, tools):
    # TODO: tools/list, tools/call, and errors
    pass
`,
        solution: `
def handle(message, tools):
    base = {"jsonrpc": "2.0", "id": message.get("id")}
    method = message.get("method")
    if method == "tools/list":
        return {**base, "result": {"tools": [{"name": n, "description": d} for n, (d, f) in tools.items()]}}
    if method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        if name not in tools:
            return {**base, "error": {"code": -32602, "message": "Unknown tool: " + str(name)}}
        try:
            value = tools[name][1](**params.get("arguments", {}))
            return {**base, "result": {"content": [{"type": "text", "text": str(value)}], "isError": False}}
        except Exception as e:
            return {**base, "result": {"content": [{"type": "text", "text": str(e)}], "isError": True}}
    return {**base, "error": {"code": -32601, "message": "Method not found"}}
`,
        hints: ['Build the common part first: {"jsonrpc": "2.0", "id": message.get("id")}.', 'Call the tool with tools[name][1](**arguments).', 'Catch exceptions from the tool and report them in the result with isError True, not as a JSON-RPC error.'],
        explain: `This split mirrors the MCP specification: problems with the request itself (an unknown method or tool) are protocol errors, while a tool that runs and fails returns its error message in the result, so the model can read it and try something else. Real servers also validate the arguments against each tool's input schema.`,
        tests: [
          { name: 'tools/list', setup: 'tools = {"add": ("Add two numbers", lambda a, b: a + b), "fail": ("Always fails", lambda: 1 / 0)}', call: '[t["name"] for t in handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, tools)["result"]["tools"]]', expect: "['add', 'fail']" },
          { name: 'a successful call', setup: 'tools = {"add": ("Add two numbers", lambda a, b: a + b), "fail": ("Always fails", lambda: 1 / 0)}', call: 'handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 2, "b": 3}}}, tools)', expect: "{'jsonrpc': '2.0', 'id': 2, 'result': {'content': [{'type': 'text', 'text': '5'}], 'isError': False}}" },
          { name: 'a failing tool reports isError', setup: 'tools = {"add": ("Add two numbers", lambda a, b: a + b), "fail": ("Always fails", lambda: 1 / 0)}', call: 'handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "fail", "arguments": {}}}, tools)["result"]["isError"]', expect: 'True' },
          { name: 'an unknown tool', setup: 'tools = {"add": ("Add two numbers", lambda a, b: a + b)}', call: 'handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}, tools)["error"]', expect: "{'code': -32602, 'message': 'Unknown tool: nope'}" },
          { name: 'an unknown method', setup: 'tools = {}', call: 'handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"}, tools)', hidden: true },
          { name: 'the error text of a failing tool', setup: 'tools = {"div": ("Divide", lambda a, b: a / b)}', call: 'handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "div", "arguments": {"a": 1, "b": 0}}}, tools)', hidden: true }
        ]
      },
      {
        id: 'tool-argument-check', title: 'Validate Tool Arguments', difficulty: 'medium',
        prompt: `Every MCP tool declares an input schema (JSON Schema). Implement \`check_arguments(schema, args)\` for a small subset of JSON Schema and return a list of problems, in this order:

1. for each name in \`schema["required"]\` (in that order) that is missing from \`args\`: \`"missing required argument: <name>"\`;
2. then for each argument in \`args\` (in its order):
   - not in \`schema["properties"]\`: \`"unexpected argument: <name>"\`;
   - wrong type: \`"<name> should be of type <type>"\`, where \`"string"\` means \`str\`, \`"integer"\` means \`int\`, \`"number"\` means \`int\` or \`float\`, and \`"boolean"\` means \`bool\`. Booleans do **not** count as integers or numbers.

An empty list means the arguments are valid.`,
        starter: `
def check_arguments(schema, args):
    # TODO
    pass
`,
        solution: `
def check_arguments(schema, args):
    problems = []
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            problems.append("missing required argument: " + name)
    allowed = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}
    for name, value in args.items():
        if name not in props:
            problems.append("unexpected argument: " + name)
            continue
        want = props[name].get("type")
        if want not in allowed:
            continue
        wrong = not isinstance(value, allowed[want]) or (want != "boolean" and isinstance(value, bool))
        if wrong:
            problems.append(name + " should be of type " + want)
    return problems
`,
        hints: ['Check required names first, then loop over the provided arguments.', 'isinstance(True, int) is True in Python, so reject booleans explicitly for integer and number.'],
        explain: `Validating arguments before running a tool turns a model's malformed call into a clear message it can fix on the next step, instead of a crash or, worse, an action taken with wrong inputs. MCP SDKs do this automatically from the schema.`,
        tests: [
          { name: 'valid arguments', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}, "required": ["city"]}', call: 'check_arguments(schema, {"city": "Oslo", "days": 3})', expect: '[]' },
          { name: 'a missing argument', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}', call: 'check_arguments(schema, {"days": 3})', expect: "['missing required argument: city']" },
          { name: 'a wrong type', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}', call: 'check_arguments(schema, {"city": "Oslo", "days": "3"})', expect: "['days should be of type integer']" },
          { name: 'booleans are not integers', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}', call: 'check_arguments(schema, {"city": "Oslo", "days": True})', expect: "['days should be of type integer']" },
          { name: 'an unexpected argument', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}', call: 'check_arguments(schema, {"city": "Oslo", "units": "C"})', hidden: true },
          { name: 'several problems in order', setup: 'schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "scale": {"type": "number"}}, "required": ["city"]}', call: 'check_arguments(schema, {"days": 2.5, "scale": 2, "x": 1})', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Model Context Protocol, specification and documentation', url: 'https://modelcontextprotocol.io' },
      { title: 'Anthropic, Introducing the Model Context Protocol (2024)', url: 'https://www.anthropic.com/news/model-context-protocol' },
      { title: 'MCP Python SDK', url: 'https://github.com/modelcontextprotocol/python-sdk' }
    ]
  },

  // =====================================================================================
  {
    id: 'discrete-math', branch: 'math', title: 'Discrete Math for ML: Sets, Counting, Graphs and Complexity', short: 'Discrete Math', level: 1, minutes: 40, prereqs: ['python-essentials'],
    tags: ['discrete math', 'sets', 'logic', 'boolean', 'de morgan', 'jaccard', 'counting', 'combinatorics', 'permutations', 'combinations', 'binomial coefficient', 'pigeonhole', 'graphs', 'bfs', 'dag', 'topological sort', 'complexity', 'big-o'],
    summary: 'The discrete side of the mathematics behind machine learning: sets and logic, and how they map onto boolean masks and metrics; counting with permutations and combinations, and why search spaces explode; graphs and breadth-first search (computation graphs, knowledge graphs, this map of prerequisites); and algorithmic complexity with big-O notation.',
    why: `Computers work with finite, discrete things: sets of tokens, counts of combinations, graphs of connections, steps of an algorithm. Discrete maths is the vocabulary for them. It tells you how many settings a grid search will try, why trying every subset of features is hopeless, how a computation graph lets backpropagation run, and why attention gets expensive on long inputs.`,
    objectives: [
      'Use set operations and logic, and translate them into boolean masks and similarity measures such as Jaccard',
      'Count permutations and combinations, and recognize combinatorial explosion in search spaces',
      'Represent graphs with adjacency lists and matrices, and find shortest paths with breadth-first search',
      'Estimate how the cost of an algorithm grows, with big-O notation'
    ],
    sections: [
      { type: 'text', title: "Why discrete maths?", md: `Most of this atlas uses continuous maths: gradients, probabilities and vectors of real numbers. But computers ultimately count and compare: tokens in a vocabulary, items in a set, edges in a graph, steps of an algorithm. Discrete maths covers that side, and it shows up throughout machine learning:

- **Sets and logic**: filters on data, boolean masks, metrics such as precision and recall, and the similarity of two sets.
- **Counting**: how many settings a grid search tries, how many n-grams a vocabulary can form, and why trying every possibility is usually hopeless.
- **Graphs**: computation graphs (which make backpropagation possible), knowledge graphs, molecules, social networks, and the prerequisite map of this app.
- **Complexity**: how the cost of an algorithm grows with the size of its input, and therefore what will still work on a million examples or a long document.` },
      { type: 'text', title: "Sets and logic", md: `A **set** is a collection of distinct items. Its basic operations, with their Python spelling for sets (and NumPy's for boolean masks):

- **intersection** A ∩ B, the items in both: \`a & b\`;
- **union** A ∪ B, the items in either or both: \`a | b\`;
- **difference** A − B, the items in A but not in B: \`a - b\` (for masks, \`m1 & ~m2\`);
- **complement**, everything not in A: \`~m\` for a mask.

Logic is the same algebra with true and false. **De Morgan's laws** say how negation moves through *and* and *or*, which is handy whenever you invert a filter:

- not (A and B) = (not A) or (not B);
- not (A or B) = (not A) and (not B).

Many machine-learning quantities are ratios of set sizes. Precision is |predicted ∩ actual| / |predicted| and recall is |predicted ∩ actual| / |actual| (see [Evaluation Metrics](#/concept/evaluation-metrics)). The **Jaccard similarity** compares two sets directly:

$$ J(A, B) = |A ∩ B| / |A ∪ B| $$
:::where
A, B — two sets, for example the distinct words of two documents
|A ∩ B| — the number of elements the two sets share
|A ∪ B| — the number of distinct elements in either set
J(A, B) — the Jaccard similarity, from 0 (nothing shared) to 1 (identical sets)
:::

It is used to find near-duplicate documents in training data (with the MinHash trick, which estimates it quickly), and, as *intersection over union*, to score predicted boxes in object detection.` },
      { type: 'code', title: "Sets, masks and De Morgan's laws", code: `import numpy as np

doc_a = set("the cat sat on the mat".split())
doc_b = set("the cat lay on the rug".split())
print("A ∩ B (in both):    ", sorted(doc_a & doc_b))
print("A ∪ B (in either):  ", sorted(doc_a | doc_b))
print("A − B (only in A):  ", sorted(doc_a - doc_b))
print(f"Jaccard similarity |A ∩ B| / |A ∪ B| = {len(doc_a & doc_b)}/{len(doc_a | doc_b)} = {len(doc_a & doc_b) / len(doc_a | doc_b):.2f}")

# The same logic on arrays: boolean masks select rows of data
age = np.array([23, 35, 47, 52, 61, 19])
member = np.array([True, False, True, True, False, False])
not_both = ~((age > 40) & member)                    # NOT (over 40 AND member)
either_not = ~(age > 40) | ~member                   # (NOT over 40) OR (NOT member)
print("not (over 40 and member):", not_both.astype(int))
print("De Morgan's law holds:", bool(np.array_equal(not_both, either_not)))`, after: `The two sentences share three words (the, cat, on) out of seven distinct words in total, so their Jaccard similarity is 3/7. Repeated words count once, because sets ignore duplicates. The masks show the same logic on data: negating "over 40 and a member" gives exactly the rows that are "not over 40, or not a member".`, predict: {"type":"mcq","q":"\"the cat sat on the mat\" and \"the cat lay on the rug\" share 3 of the 7 distinct words they contain between them. What is their Jaccard similarity?","options":["0.30","0.43","0.60","0.75"],"answer":1,"explain":"3 shared words divided by 7 distinct words in total: 3/7 ≈ 0.43.","expect":["Jaccard similarity |A ∩ B| / |A ∪ B| = 3/7 = 0.43"]} },
      { type: 'text', title: "Counting: the rule of product, permutations and combinations", md: `- **The rule of product:** if one choice has a options and another has b, together they have a × b. A grid search over 4 learning rates, 3 batch sizes, 4 depths and 3 dropout rates trains 4 × 3 × 4 × 3 = 144 models.
- **Permutations** count ordered arrangements, **combinations** unordered selections:

$$ P(n, k) = n! / (n − k)!          C(n, k) = n! / (k! · (n − k)!) $$
:::where
n — the number of items to choose from
k — how many items are chosen
n! — "n factorial": n × (n − 1) × … × 1, the number of ways to put n items in order
P(n, k) — the number of ordered selections of k items out of n
C(n, k) — "n choose k": the number of unordered selections, the same binomial coefficient as in the binomial distribution
:::

- **All subsets:** each of n items is either in or out, so there are 2ⁿ subsets. That is why choosing features by trying every subset is hopeless beyond a few dozen features, and why feature selection uses greedy or regularized methods instead (see [Regularization](#/concept/regularization)).
- **Sequences:** a vocabulary of V tokens can form Vⁿ different n-grams, which is why counting-based language models run out of data so quickly (see [Language Models](#/concept/language-models)).
- **The pigeonhole principle:** putting more items into fewer boxes forces at least two items into one box. Hashing a million distinct words into 100,000 buckets must cause collisions, a trade-off that the "hashing trick" for features accepts on purpose.
- **Probability by counting:** when all outcomes are equally likely, a probability is favourable outcomes divided by all outcomes (see [Probability & Distributions](#/concept/probability)).` },
      { type: 'code', title: "How fast do counts explode?", code: `import math

grid = {"learning rate": [1e-4, 3e-4, 1e-3, 3e-3], "batch size": [16, 32, 64], "layers": [2, 4, 6, 8], "dropout": [0.0, 0.1, 0.3]}
size = math.prod(len(v) for v in grid.values())
print(f"grid search: {' x '.join(str(len(v)) for v in grid.values())} = {size} training runs")
print(f"choosing 5 of 30 features: C(30, 5) = {math.comb(30, 5):,} subsets")
print(f"every subset of 30 features: 2^30 = {2 ** 30:,}")
print(f"orders of 15 tasks: 15! = {math.factorial(15):,}")
for name, count in (("2^30 subsets", 2 ** 30), ("2^60 subsets", 2 ** 60), ("20! orderings", math.factorial(20))):
    seconds = count / 1e9                                  # checking a billion possibilities per second
    print(f"checking {name} at a billion per second: {seconds:,.1f} s = {seconds / 3.15e7:,.1f} years")
print("2^30 is over a billion:", 2 ** 30 > 1e9)`, after: `Thirty features already have over a billion subsets, which a fast computer could just about check in a second if each check took a nanosecond. But a real check means training a model, and sixty features would take decades even at that impossible speed. Exponential (2ⁿ) and factorial (n!) growth outrun any hardware, so algorithms must avoid enumerating such spaces: greedy search, random search, gradient-based optimization and dynamic programming all exist for this reason.`, predict: {"type":"mcq","q":"How many different subsets can be formed from 30 features, counting the empty set and the set of all 30?","options":["900","About 30,000","About a billion","About 10³⁰"],"answer":2,"explain":"Each feature is either in or out: 2³⁰ = 1,073,741,824 subsets, just over a billion.","expect":["2^30 is over a billion: True"]} },
      { type: 'chart', title: "How counts grow with n", after: `On a log scale polynomial growth flattens out while exponential growth is a straight line and factorial growth curves upwards. By n = 20 there are about a million subsets and over 10¹⁸ orderings, against only 400 pairs: algorithms that enumerate subsets or orderings stop being usable long before the input gets large.`, chart: {"kind":"line","x":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"yLog":true,"series":[{"name":"n log₂ n (sorting)","y":[2,4.755,8,11.61,15.51,19.65,24,28.53,33.22,38.05,43.02,48.11,53.3,58.6,64,69.49,75.06,80.71,86.44]},{"name":"n² (all pairs)","y":[4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400]},{"name":"2ⁿ (all subsets)","y":[4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576]},{"name":"n! (all orderings)","y":[2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800,87178291200,1307674368000,20922789888000,355687428096000,6402373705728000,121645100408832000,2432902008176640000]}],"xLabel":"n","yLabel":"count (log scale)"} },
      { type: 'text', title: "Graphs", md: `A **graph** consists of **vertices** (nodes) joined by **edges**. Edges can be directed (from A to B) or undirected, and can carry weights such as distances or strengths. Graphs are everywhere in machine learning: molecules (atoms and bonds), social and citation networks, road maps, knowledge graphs, and the **computation graphs** that autograd builds to run backpropagation (see [Autograd](#/concept/autograd)).

Two ways to store a graph:

- an **adjacency list**: for each node, the list of its neighbours; compact for sparse graphs, and the usual choice in code;
- an **adjacency matrix** A: an n × n table with A[i][j] = 1 when there is an edge from i to j. It turns graph operations into linear algebra: the entries of A² count paths of length 2, and graph neural networks mix each node's features with its neighbours' by multiplying by A (see [Graph Neural Networks](#/concept/gnn)).

Important ideas:

- **Paths and shortest paths.** Breadth-first search (BFS) explores a graph in rings of increasing distance, so the first time it reaches a node is along a shortest path (counting edges). Dijkstra's algorithm does the same when edges have weights.
- **Trees** are connected graphs without cycles: decision trees, parse trees, the hierarchy of clusters.
- **DAGs** (directed acyclic graphs) have directed edges and no cycles: computation graphs, task dependencies, and this app's map of prerequisites. A DAG always has a **topological order**, a list in which every node comes after all the nodes that point to it: a valid order in which to learn, compute or build.` },
      { type: 'code', title: "Shortest prerequisite chains with breadth-first search", code: `from collections import deque

builds_on = {   # concept -> the concepts that build directly on it (a small slice of a map like this app's)
    "vectors": ["matrices", "embeddings"], "derivatives": ["gradient descent"], "matrices": ["neural networks"],
    "gradient descent": ["neural networks"], "neural networks": ["backpropagation", "embeddings"],
    "backpropagation": ["transformers"], "embeddings": ["attention"], "attention": ["transformers"], "transformers": [],
}

def bfs(graph, start):                               # distances in edges from start, ring by ring
    dist, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist
print("steps from 'vectors':", bfs(builds_on, "vectors"))

def topological_order(graph):                        # Kahn's algorithm: repeatedly take a node nothing still points to
    indegree = {n: 0 for n in graph}
    for n in graph:
        for m in graph[n]:
            indegree[m] += 1
    ready = deque(sorted(n for n, d in indegree.items() if d == 0))
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for m in graph[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                ready.append(m)
    return order

order = topological_order(builds_on)
print("a valid learning order:", " -> ".join(order))
print("every concept comes after what it builds on:", all(order.index(a) < order.index(b) for a in builds_on for b in builds_on[a]))`, after: `BFS finds that transformers are three steps from vectors (vectors → embeddings → attention → transformers), even though the longer route through matrices, neural networks and backpropagation exists too. Kahn's algorithm produces an order in which every concept appears after everything it builds on; if the graph had a cycle, some nodes would never become ready, which is also how cycles are detected. This app orders its learning tracks the same way.`, predict: {"type":"mcq","q":"Following the arrows, what is the smallest number of steps from \"vectors\" to \"transformers\"?","options":["2","3","4","5"],"answer":1,"explain":"vectors → embeddings → attention → transformers: 3 steps. The route through matrices, neural networks and backpropagation is 4 steps long.","expect":["'transformers': 3"]} },
      { type: 'text', title: "Algorithmic complexity and big-O", md: `To compare algorithms independently of the computer they run on, count how their cost **grows** with the size of the input:

$$ f(n) = O(g(n))  ⇔  f(n) ≤ c · g(n) for some constant c and all large enough n $$
:::where
n — the size of the input: examples, tokens, features or nodes
f(n) — the cost of the algorithm: its number of operations, its running time or its memory
g(n) — a simple reference function such as n, n log n or n²
c — a constant that does not depend on n
O — "big O": an upper bound on the growth of f, ignoring constant factors
:::

Common growth rates, with examples from machine learning:

- **O(log n)**: binary search in sorted data, the depth of a balanced tree.
- **O(n)**: one pass over the data, such as computing a mean.
- **O(n log n)**: sorting, which is why ranking metrics such as ROC AUC are cheap to compute.
- **O(n · d)**: one k-nearest-neighbour query over n training points with d features, or one epoch of gradient descent for a linear model. Approximate vector indexes exist to beat it (see [Vector Search](#/concept/vector-search)).
- **O(n²)**: comparing all pairs, as in self-attention over n tokens or a full distance matrix (see [Efficient Attention](#/concept/efficient-attention)).
- **O(n³)**: naive matrix multiplication and matrix inversion, which limits exact Gaussian processes to a few thousand points (see [Gaussian Processes](#/concept/gaussian-processes)).
- **O(2ⁿ)** and **O(n!)**: all subsets and all orderings, as above.

Big-O ignores constant factors, which still matter in practice: a GPU can make an O(n²) algorithm faster than an O(n log n) one for the sizes you actually have. Use big-O to predict how things scale, and measure to decide.` },
      { type: 'code', title: "What doubling the input does", code: `import math
print("     n   linear scan   binary search   sorting (n log n)   all pairs (n²)")
rows = {}
for n in (1_000, 2_000, 4_000, 8_000):
    rows[n] = (n, math.ceil(math.log2(n)), round(n * math.log2(n)), n * n)
    print(f"{n:6,} {rows[n][0]:13,} {rows[n][1]:15,} {rows[n][2]:19,} {rows[n][3]:16,}")
print("doubling n multiplies the pairs by", rows[2_000][3] // rows[1_000][3], "and the binary-search steps by about", round(rows[2_000][1] / rows[1_000][1], 2))
print("all pairs grow 4x per doubling:", rows[2_000][3] == 4 * rows[1_000][3])`, after: `Doubling the input doubles a linear scan, adds a single step to binary search, slightly more than doubles sorting, and quadruples anything that compares all pairs. Eight thousand tokens already mean 64 million pairs in every attention layer, which is why long contexts are expensive and why so much research goes into cheaper attention.`, predict: {"type":"mcq","q":"Self-attention compares every token with every other. When the input grows from 1,000 to 2,000 tokens, the number of comparisons…","options":["doubles","quadruples","grows by about 10%","stays the same"],"answer":1,"explain":"All pairs: (2n)² = 4n². Doubling the length quadruples the comparisons.","expect":["all pairs grow 4x per doubling: True"]} },
      { type: 'check', question: {"type":"tf","q":"If algorithm A is O(n log n) and algorithm B is O(n²), A is always faster than B.","answer":false,"explain":"Big-O describes growth for large n and ignores constants. For small inputs, or with a much faster implementation, B can win; A wins eventually as n grows."} }
    ],
    when: `- Sizing a search: grid search, feature subsets, n-gram vocabularies.
- Modelling relationships: social networks, molecules, knowledge graphs, dependency graphs (see Graph Neural Networks).
- Predicting how a method scales with data size, sequence length or dimension before running it.`,
    pitfalls: `- Underestimating combinatorial growth: 2^d feature subsets, Vⁿ possible n-grams.
- Confusing "and" with "or" in filters; De Morgan's laws help: not (A and B) = (not A) or (not B).
- Judging an algorithm by its speed on tiny inputs: big-O describes how the cost grows.`,
    takeaways: [
      'Sets and logic map directly onto boolean masks; Jaccard similarity is the size of the overlap of two sets divided by the size of their union.',
      'The rule of product counts grids, combinations C(n, k) count subsets, and growth like 2ⁿ or n! makes exhaustive search impossible.',
      'Graphs (nodes and edges) describe computations, molecules and knowledge; breadth-first search finds shortest paths in unweighted graphs.',
      'Big-O describes how cost grows with input size: O(n²) attention quadruples when the sequence doubles.'
    ],
    quiz: [
      {"type":"fill","q":"How many ways are there to choose 3 items out of 10 when order does not matter, C(10, 3)?","answer":["120"],"explain":"C(10, 3) = 10! / (3! · 7!) = (10 × 9 × 8) / (3 × 2 × 1) = 120."},
      {"type":"mcq","q":"By De Morgan's laws, \"not (A and B)\" is the same as…","options":["(not A) and (not B)","(not A) or (not B)","A or B","not A"],"answer":1,"explain":"Negation turns and into or: something fails \"A and B\" when it fails A or fails B."},
      {"type":"mcq","q":"Which algorithm finds the shortest path, counted in edges, in an unweighted graph?","options":["Depth-first search","Breadth-first search","Sorting the nodes","Gradient descent"],"answer":1,"explain":"BFS visits nodes in order of their distance from the start, so it reaches each node first along a shortest path."},
      {"type":"tf","q":"A directed graph that contains a cycle has a topological order.","answer":false,"explain":"In a cycle every node comes after another node of the cycle, so no order can put each node after all of its predecessors."},
      {"type":"mcq","q":"For large n, which of these grows fastest?","options":["n²","n³","n log n","2ⁿ"],"answer":3,"explain":"Exponential growth eventually overtakes every polynomial: 2ⁿ passes n³ at n = 10 and leaves it far behind."},
      {"type":"fill","q":"A grid search tries 5 learning rates, 4 batch sizes and 3 random seeds. How many training runs is that?","answer":["60"],"explain":"By the rule of product, 5 × 4 × 3 = 60 runs."}
    ],
    problems: [
      {
        id: 'jaccard-similarity', title: 'Jaccard Similarity', difficulty: 'easy',
        prompt: `The **Jaccard similarity** of two collections is the size of the intersection of their sets divided by the size of their union. Implement \`jaccard(a, b)\` for any two iterables (duplicates are ignored). By convention, two empty collections have similarity 1.0.

\`\`\`text
jaccard("the cat sat".split(), "the cat ran".split())   ->  0.5      {the, cat} / {the, cat, sat, ran}
jaccard([1, 2], [3])                                    ->  0.0
\`\`\``,
        starter: `
def jaccard(a, b):
    # TODO
    pass
`,
        solution: `
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)
`,
        hints: ['Convert both inputs to sets first.', 'A & B is the intersection and A | B the union.'],
        explain: `Jaccard similarity is used for near-duplicate detection (comparing sets of word n-grams, as in MinHash deduplication of training data), for comparing predicted and true sets of labels, and as intersection over union (IoU) for regions in object detection.`,
        tests: [
          { call: 'jaccard("the cat sat".split(), "the cat ran".split())', expect: '0.5' },
          { name: 'disjoint', call: 'jaccard([1, 2], [3])', expect: '0.0' },
          { name: 'both empty', call: 'jaccard([], [])', expect: '1.0' },
          { name: 'duplicates are ignored', call: 'jaccard([1, 1, 2], [1])', expect: '0.5' },
          { name: 'character sets', call: 'jaccard("banana", "bandana")', hidden: true }
        ]
      },
      {
        id: 'bfs-distances', title: 'Shortest Paths with Breadth-First Search', difficulty: 'medium',
        prompt: `A directed graph is given as a dict that maps each node to the list of nodes its edges point to (a node may appear only as a neighbour). Implement \`bfs_distances(graph, start)\` that returns a dict mapping every node reachable from \`start\` to the number of edges on its shortest path from \`start\` (\`start\` itself has distance 0). Use breadth-first search: visit nodes in order of their distance with a queue (\`collections.deque\`).

\`\`\`text
graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
bfs_distances(graph, "a")   ->  {"a": 0, "b": 1, "c": 1, "d": 2}
\`\`\``,
        starter: `
from collections import deque

def bfs_distances(graph, start):
    # TODO
    pass
`,
        solution: `
from collections import deque

def bfs_distances(graph, start):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist
`,
        hints: ['Record a node\'s distance when you first discover it, and never update it again.', 'graph.get(node, []) handles nodes that have no outgoing edges listed.'],
        explain: `Breadth-first search explores the graph in rings of increasing distance, so the first time it reaches a node is along a shortest path. It runs in O(V + E) time. The same idea finds the shortest chain of prerequisites on this app's concept map.`,
        tests: [
          { call: 'bfs_distances({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, "a")', expect: "{'a': 0, 'b': 1, 'c': 1, 'd': 2}" },
          { name: 'unreachable nodes are left out', call: 'bfs_distances({"a": ["b"], "b": [], "c": ["a"]}, "a")', expect: "{'a': 0, 'b': 1}" },
          { name: 'a node that only appears as a neighbour', call: 'bfs_distances({"a": ["b"]}, "a")', expect: "{'a': 0, 'b': 1}" },
          { name: 'cycles', call: 'bfs_distances({"a": ["b"], "b": ["c"], "c": ["a", "d"]}, "a")', expect: "{'a': 0, 'b': 1, 'c': 2, 'd': 3}" },
          { name: 'a larger graph vs reference', setup: 'import random\nrandom.seed(11)\ng = {i: random.sample(range(30), 3) for i in range(30)}', call: 'bfs_distances(g, 0)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'MIT OpenCourseWare, Mathematics for Computer Science (6.042J)', url: 'https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/' },
      { title: 'Python documentation, itertools (permutations, combinations, product)', url: 'https://docs.python.org/3/library/itertools.html' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
