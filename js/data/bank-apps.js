/* ML Atlas – extra quiz questions for the Applications lessons. They appear only in custom quizzes (#/quiz), never in a
   lesson's own quiz or the review queue. Keyed by lesson id; the format is the lesson quiz format (docs/CONTENT_GUIDE.md). */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});
  MLA.data = MLA.data || {};
  MLA.data.quizBank = MLA.data.quizBank || {};
  Object.assign(MLA.data.quizBank, {
    'prompting': [
      { type: 'mcq', q: "What does a system prompt usually hold?",
        options: ["Standing instructions for the whole conversation, such as role, rules and output format", "The user's latest question", "The model's weights", "The previous answer"], answer: 0,
        explain: "It applies to every turn, while user and assistant messages alternate after it." },
      { type: 'mcq', q: "You need replies as JSON with the fields name and age. What is the most reliable approach?",
        options: ["Give the exact schema and an example, then parse and validate the reply and retry on failure, or use the API's structured-output mode", "Ask politely for JSON", "Raise the temperature", "Trust whatever comes back"], answer: 0,
        explain: "Models usually follow a clear format, but code that consumes the output must still check it. Structured-output modes constrain decoding to the schema." },
      { type: 'tf', q: "Asking a model to work through the steps before answering helps most on multi-step problems such as arithmetic word problems.", answer: true,
        explain: "Writing intermediate results gives the model room to compute. On simple lookups it mainly adds tokens and latency." },
      { type: 'mcq', q: "A prompt wraps a document in tags and then says: summarise the document in 3 bullet points for a manager. What do the tags help with?",
        options: ["Separating the material from the instructions, so the model knows what to summarise and what to follow", "Making the prompt shorter", "Encrypting the document", "Choosing a different model"], answer: 0,
        explain: "Clear delimiters reduce confusion between data and instructions. They do not make injected instructions inside the document harmless." }
    ],
    'rag': [
      { type: 'mcq', q: "What is a risk of very large chunks?",
        options: ["Each chunk mixes several topics, so it matches queries poorly and fills the prompt with irrelevant text", "They cannot be embedded", "They make retrieval exact", "There is no risk"], answer: 0,
        explain: "One vector has to summarise the whole chunk. Small chunks have the opposite problem: they can lose the context a passage needs." },
      { type: 'mcq', q: "What does a reranker do in a RAG pipeline?",
        options: ["Reorders the retrieved candidates with a more accurate model before the best few go into the prompt", "Generates the answer", "Splits the documents", "Embeds the query"], answer: 0,
        explain: "Fast retrieval finds, say, 50 candidates; a slower cross-encoder scores each against the question, and only the top 5 are used." },
      { type: 'tf', q: "Asking the model to cite the chunk that supports each claim makes its answers easier to check.", answer: true,
        explain: "A citation can be verified against the retrieved text, and a claim without one is a warning sign." },
      { type: 'mcq', q: "Why do chunks overlap?",
        options: ["A passage that straddles a chunk boundary still appears whole in at least one chunk", "It makes the index smaller", "It removes duplicates", "It speeds up embedding"], answer: 0,
        explain: "Without overlap, a key sentence cut in half might be found by neither chunk. The cost is some duplicated storage." }
    ],
    'agents': [
      { type: 'mcq', q: "In tool calling, who actually runs the tool?",
        options: ["Your application code, which then sends the result back to the model", "The model itself", "The user", "The model provider, automatically"], answer: 0,
        explain: "The model only emits a structured request. Your code decides whether to run it, runs it, and returns the result as a message." },
      { type: 'mcq', q: "What should an agent do before a destructive action, such as deleting files?",
        options: ["Ask a person to confirm", "Go ahead silently", "Retry it several times", "Leave it out of the logs"], answer: 0,
        explain: "Irreversible actions deserve a human check, since a misunderstanding or an injected instruction could otherwise cause real damage." },
      { type: 'tf', q: "Giving an agent only the tools and permissions its task needs limits the damage a prompt injection can do.", answer: true,
        explain: "An injected instruction can only use what the agent has. Least privilege turns a possible disaster into a contained failure." },
      { type: 'mcq', q: "What does MCP give an agent application?",
        options: ["A standard way to connect to tools and data offered by separate servers", "A new model architecture", "A tokenizer", "A training algorithm"], answer: 0,
        explain: "With one protocol, an application can use any compatible server instead of writing a custom integration for each tool." }
    ],
    'llm-evaluation': [
      { type: 'mcq', q: "What is token-level F1 between a prediction and a reference answer?",
        options: ["The harmonic mean of the share of predicted tokens found in the reference and the share of reference tokens found in the prediction", "Exact match of the two strings", "The number of tokens in the prediction", "The judge's score out of 1"], answer: 0,
        explain: "It gives partial credit when the answer is close but not identical, unlike exact match." },
      { type: 'fill', q: "The prediction 'the red car' and the reference 'a red car' share 2 tokens, so precision and recall are both 2/3. What is the token F1?",
        answer: ["2/3", "/^0?[.]6[67][0-9]*$/"], display: "2/3 (about 0.67)",
        explain: "F1 = 2PR / (P + R). When P = R, F1 equals them both: 2/3." },
      { type: 'mcq', q: "How do you check whether an LLM judge can be trusted?",
        options: ["Compare its verdicts with human labels on a sample and measure the agreement", "Ask the judge whether it is reliable", "Switch to a bigger judge without checking", "Read a few of its outputs"], answer: 0,
        explain: "Agreement beyond chance, measured on cases people labelled, tells you how far to rely on the judge and where it fails." },
      { type: 'mcq', q: "A bug that was fixed last month appears again. What would have caught it?",
        options: ["Adding the failing case to the eval set as a regression test", "A bigger model", "A higher temperature", "Fewer test cases"], answer: 0,
        explain: "Every failure found in production is a ready-made test case. Running the eval set on each change keeps old bugs from returning." }
    ],
    'vector-search': [
      { type: 'mcq', q: "In an HNSW index, what does raising the search breadth (ef_search) do?",
        options: ["Raises recall but makes each query slower", "Uses less memory", "Makes queries faster and more accurate at once", "Changes the stored vectors"], answer: 0,
        explain: "A wider search explores more of the graph, finding more true neighbours at the cost of more distance computations." },
      { type: 'mcq', q: "Reciprocal rank fusion scores a document Σ 1/(k + rank) over the retrievers. With k = 60, what does a document ranked 1st by one retriever and missing from the other score?",
        options: ["1/61", "1/60", "1", "60"], answer: 0,
        explain: "Only the retriever that found it contributes: 1/(60 + 1). Documents ranked well by both retrievers collect two such terms." },
      { type: 'tf', q: "For unit vectors, ranking by Euclidean distance gives the same order as ranking by cosine similarity.", answer: true,
        explain: "For unit vectors ‖a − b‖² = 2 − 2·cos(a, b), so a smaller distance always means a larger cosine." },
      { type: 'mcq', q: "About how much work does exact (flat) search over N vectors of dimension d take per query?",
        options: ["N · d multiply-adds", "log N operations", "d² operations", "N² operations"], answer: 0,
        explain: "Every stored vector is compared with the query. Approximate indexes avoid most of those comparisons." }
    ],
    'advanced-agents': [
      { type: 'mcq', q: "What is just-in-time retrieval for an agent's context?",
        options: ["Fetching files or data with tools when they are needed, instead of loading everything up front", "Retrieving the whole codebase at the start", "Retraining the model during the task", "Caching every answer"], answer: 0,
        explain: "The context holds only what the current step needs, which keeps it small and focused. The agent keeps references, such as file paths, to fetch more later." },
      { type: 'mcq', q: "What does a plan-and-execute agent do?",
        options: ["Writes a plan first, then carries out the steps, revising the plan when results surprise it", "Acts with no plan at all", "Runs many agents in parallel", "Only answers questions"], answer: 0,
        explain: "An explicit plan keeps long tasks on track and makes progress easy to check, at the cost of an extra planning call." },
      { type: 'tf', q: "A multi-agent design usually uses more tokens than a single agent doing the same task.", answer: true,
        explain: "Each agent needs its own context, and the agents exchange messages. The design pays off mainly for broad tasks that split into independent parts." },
      { type: 'mcq', q: "Why let an agent keep a notes file during a long task?",
        options: ["Important facts survive when older context is compacted or dropped", "It slows the agent down on purpose", "It trains the model", "Tools require it"], answer: 0,
        explain: "Notes written outside the context window act as memory the agent can read back after compaction or in a new session." }
    ],
    'ai-engineering': [
      { type: 'mcq', q: "A request fails with a rate-limit error (HTTP 429). What is the right reaction?",
        options: ["Retry after a delay that grows with each attempt, with some random jitter", "Retry at once in a tight loop", "Give up after the first error", "Switch to another person's API key"], answer: 0,
        explain: "Exponential backoff gives the service time to recover, and jitter stops many clients from retrying at the same moment." },
      { type: 'mcq', q: "Why stream a model's response to the user?",
        options: ["Text appears as it is generated, so the wait feels much shorter", "It makes generation cheaper", "It makes answers more accurate", "It removes the context limit"], answer: 0,
        explain: "The total time is the same, but the first words arrive after the time to first token instead of after the whole answer." },
      { type: 'tf', q: "Cost per successful outcome can favour a pricier model, if a cheaper one fails more often and needs retries or human fixes.", answer: true,
        explain: "What matters is the full cost of getting a good result, not the price per token alone." },
      { type: 'mcq', q: "What is the best way to choose between two models for your task?",
        options: ["Run both on your own eval set and compare quality, cost and latency", "Pick the one with the higher public benchmark score", "Pick the newest", "Pick the largest"], answer: 0,
        explain: "Public benchmarks may not resemble your task. A few dozen to a few hundred of your own cases settle the question." }
    ],
    'llm-observability': [
      { type: 'mcq', q: "Which data should be kept out of logs and traces, or masked?",
        options: ["Personal data and secrets that appear in prompts and outputs", "Token counts", "Latency", "The model version"], answer: 0,
        explain: "Traces are read by many people and kept for a long time. Mask or drop personal data, and keep retention short." },
      { type: 'mcq', q: "What is a guardrail in an LLM application?",
        options: ["A check on inputs or outputs, such as blocking leaked secrets or off-topic requests, that runs alongside the model", "A bigger model", "A rate limit only", "A training dataset"], answer: 0,
        explain: "Guardrails catch problems the model itself might miss, and they log what they block so the patterns can be studied." },
      { type: 'tf', q: "Each traced LLM call should record the model, the token counts and the latency, so cost and speed can be broken down later.", answer: true,
        explain: "With those fields you can find which feature, prompt or model version drives cost or slowness." },
      { type: 'mcq', q: "What does sending a sample of production traces to human review give you?",
        options: ["A steady check on quality that automatic metrics may miss", "Faster responses", "Lower costs", "Automatic training data"], answer: 0,
        explain: "Reviewers find new failure types, which then become automatic checks and eval cases." }
    ],
    'mcp': [
      { type: 'mcq', q: "Which MCP request asks a server which tools it offers?",
        options: ["tools/list", "tools/call", "initialize", "resources/read"], answer: 0,
        explain: "The client lists the tools, with their names, descriptions and input schemas, then uses tools/call to run one." },
      { type: 'match', q: "Match each MCP server primitive with what it provides.",
        pairs: [["Tools", "Actions the model can call, such as creating an issue"], ["Resources", "Data the application can read, such as a file's contents"], ["Prompts", "Reusable prompt templates a user can pick"]],
        explain: "Tools are chosen by the model, resources by the application, and prompts by the user." },
      { type: 'tf', q: "An MCP session starts with an initialize exchange, in which client and server agree on a protocol version and their capabilities.", answer: true,
        explain: "Only after initialization do the regular requests, such as tools/list, begin." },
      { type: 'mcq', q: "Why review an MCP server before connecting it?",
        options: ["It runs code, and its tool descriptions reach the model, so a malicious server can steal data or plant instructions", "It might be slow", "It speaks JSON", "Servers can never be updated"], answer: 0,
        explain: "Connecting a server is like installing software. Prefer trusted sources, pin versions and grant the fewest permissions possible." }
    ],
    'capstone-rag': [
      { type: 'mcq', q: "Why write the evaluation set before building the RAG system?",
        options: ["So design choices are measured on fixed questions, not tuned to examples you happened to try", "It is faster to write", "The model needs it for training", "It replaces the documents"], answer: 0,
        explain: "Questions written after seeing the system's behaviour tend to favour what it already does well." },
      { type: 'mcq', q: "What does BM25 mainly rank documents by?",
        options: ["How often the query's terms occur in them, weighted by how rare the terms are and adjusted for length", "Embedding similarity", "Document length only", "Publication date"], answer: 0,
        explain: "It is a refined TF-IDF. It excels at exact names and codes but misses paraphrases with different words." },
      { type: 'tf', q: "Measuring retrieval (was the gold document found?) separately from generation (was the answer right?) shows which stage to fix.", answer: true,
        explain: "A wrong answer with the right document retrieved points at generation. A missing document points at retrieval." },
      { type: 'mcq', q: "A question's answer spans two documents, but only one of them is retrieved. Which fix helps most directly?",
        options: ["Retrieve more candidates or rerank, and test on such multi-document questions", "A larger generation model", "A lower temperature", "Removing the citations"], answer: 0,
        explain: "The generator cannot use a document it never sees. The fix belongs in retrieval, measured on questions of this type." }
    ]
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
