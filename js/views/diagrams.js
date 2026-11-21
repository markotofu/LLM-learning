/* ML Atlas – static, theme-aware SVG diagrams embedded in lessons as { type: 'demo', demo: 'dg-<name>' } sections.
 * Colours come from CSS classes (.diagram .dg-*) so they follow the dark, darker and light themes.
 * Layout is in a fixed viewBox (820 wide); on narrow screens the diagram scrolls sideways instead of shrinking. */
(function (global) {
  const MLA = global.MLA;
  MLA.components = MLA.components || {};
  const C = MLA.components;

  const DEFS = '<defs><marker id="dg-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dg-head"/></marker></defs>';

  // A rounded box with one or more bold lines and an optional small grey line, centred vertically.
  function box(x, y, w, h, cls, lines, small) {
    const cx = x + w / 2;
    const n = lines.length + (small ? 1 : 0);
    const top = y + h / 2 - (n - 1) * 9 + 5;
    let out = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="8" class="' + cls + '"/>';
    lines.forEach((t, i) => { out += '<text x="' + cx + '" y="' + (top + i * 18) + '" text-anchor="middle" class="dg-t">' + t + '</text>'; });
    if (small) out += '<text x="' + cx + '" y="' + (top + lines.length * 18) + '" text-anchor="middle" class="dg-s">' + small + '</text>';
    return out;
  }
  const arrow = (d, cls) => '<path d="' + d + '" class="dg-line' + (cls ? ' ' + cls : '') + '" marker-end="url(#dg-arr)"/>';
  const label = (x, y, t, cls) => '<text x="' + x + '" y="' + y + '" text-anchor="middle" class="' + (cls || 'dg-s') + '">' + t + '</text>';
  const plus = (x, y) => '<circle cx="' + x + '" cy="' + y + '" r="14" class="dg-box"/>' + label(x, y + 5, '+', 'dg-t');

  function diagram(name, viewBox, ariaLabel, body, caption) {
    C['demo-dg-' + name] = {
      template: '<div class="card demo-card"><div class="diagram-wrap"><svg class="diagram" viewBox="' + viewBox + '" role="img" aria-label="' + ariaLabel + '"><title>' + ariaLabel + '</title>' + DEFS + body +
        '</svg></div><p class="tiny dim diagram-hint">Scroll sideways to see the whole diagram.</p><p class="small muted">' + caption + '</p></div>'
    };
  }

  // 1. The transformer block (pre-norm)
  diagram('transformer', '0 0 820 230', 'Diagram of a pre-norm transformer block',
    box(10, 100, 100, 60, 'dg-box', ['token', 'vectors x'], 'T × d_model') +
    arrow('M110 130 H138') + box(140, 100, 80, 60, 'dg-pur', ['LayerNorm']) +
    arrow('M220 130 H243') + box(245, 95, 130, 70, 'dg-acc', ['Multi-head', 'attention']) +
    arrow('M375 130 H396') + plus(412, 130) +
    arrow('M426 130 H448') + box(450, 100, 80, 60, 'dg-pur', ['LayerNorm']) +
    arrow('M530 130 H553') + box(555, 95, 120, 70, 'dg-grn', ['Feed-forward', 'network (MLP)']) +
    arrow('M675 130 H696') + plus(712, 130) +
    arrow('M726 130 H766') + label(794, 122, 'next', 'dg-t') + label(794, 140, 'block', 'dg-t') +
    arrow('M60 100 V40 H412 V114', 'acc') + label(236, 32, 'residual stream: the input is added back after the sub-layer') +
    arrow('M440 130 V62 H712 V114', 'acc') + label(576, 54, 'residual again') +
    label(180, 182, 'normalize') + label(310, 186, 'tokens exchange information') +
    label(490, 182, 'normalize') + label(615, 186, 'each token processed on its own') +
    label(410, 220, 'Stack this block N times (12 in GPT-2 small, around 100 in the largest models).'),
    "A pre-norm transformer block. The residual stream (blue) carries every token's vector straight through; the attention and feed-forward sub-layers each read a normalized copy and add their result back. Attention is the only place where positions exchange information.");

  // 2. The RAG pipeline
  diagram('rag', '0 0 820 290', 'Diagram of a retrieval-augmented generation pipeline',
    label(78, 24, 'INDEXING (offline)', 'dg-s dg-b') +
    box(10, 36, 120, 50, 'dg-box', ['Documents'], 'cleaned text') + arrow('M130 61 H163') +
    box(165, 36, 120, 50, 'dg-box', ['Split into', 'chunks']) + arrow('M285 61 H318') +
    box(320, 36, 130, 50, 'dg-box', ['Embed chunks'], 'one vector per chunk') + arrow('M450 61 H483') +
    box(485, 36, 160, 50, 'dg-amb', ['Vector index'], 'vectors + text + metadata') +
    label(96, 154, 'ANSWERING (every question)', 'dg-s dg-b') +
    box(10, 166, 100, 60, 'dg-box', ['Question']) + arrow('M110 196 H133') +
    box(135, 166, 110, 60, 'dg-box', ['Embed', 'the query']) + arrow('M245 196 H268') +
    box(270, 166, 140, 60, 'dg-acc', ['Retrieve top-k'], 'most similar chunks') + arrow('M410 196 H433') +
    box(435, 166, 140, 60, 'dg-box', ['Prompt: question', '+ sources [1]…[k]']) + arrow('M575 196 H598') +
    box(600, 166, 80, 60, 'dg-pur', ['LLM']) + arrow('M680 196 H703') +
    box(705, 166, 110, 60, 'dg-grn', ['Answer with', 'citations']) +
    arrow('M550 86 V130 H340 V164', 'dash') + label(445, 122, 'search the index') +
    label(410, 270, 'Evaluate both halves: retrieval (recall@k, MRR) and generation (faithfulness, correctness, citation accuracy).'),
    'Retrieval-augmented generation. Documents are cleaned, chunked, embedded and indexed once; each question is embedded, the most similar chunks are retrieved (in practice often combining keyword and vector search, then reranking), and the LLM answers from those numbered sources, citing them.');

  // 3. The agent loop
  diagram('agent', '0 0 820 300', 'Diagram of an LLM agent loop',
    box(230, 8, 360, 58, 'dg-box', ['Context window (messages)'], 'system prompt · tools · notes · history') +
    box(10, 120, 120, 60, 'dg-box', ['User task']) + arrow('M130 150 H228') +
    arrow('M305 66 V118') +
    box(230, 120, 150, 60, 'dg-pur', ['Model (LLM)'], 'chooses the next action') +
    box(470, 120, 150, 60, 'dg-acc', ['Tools'], 'search · code · files · APIs') +
    arrow('M380 135 C 410 104, 440 104, 468 135') + label(425, 100, 'tool call') +
    arrow('M470 166 C 440 196, 410 196, 382 166') + label(425, 214, 'result (observation)') +
    label(718, 146, 'repeat until done', 'dg-t') + label(718, 164, 'or the step limit is hit') +
    arrow('M305 180 V226') + label(328, 210, 'done') +
    box(230, 228, 150, 52, 'dg-grn', ['Final answer'], 'no more tool calls') +
    box(470, 228, 340, 52, 'dg-red', ['Guardrails'], 'least privilege · confirmation · limits · sandbox') +
    arrow('M545 228 V183', 'dash') + label(614, 206, 'checks every call'),
    "The agent loop. The model reads the conversation so far and either answers or calls a tool; the tool's result is appended to the messages for the next step. Guardrails sit between the model and the world: least-privilege tools, confirmation before irreversible actions, step and cost limits, and sandboxes.");

  // 4. Diffusion: training and sampling
  diagram('diffusion', '0 0 820 320', 'Diagram of diffusion model training and sampling',
    label(44, 24, 'TRAINING', 'dg-s dg-b') +
    box(10, 36, 120, 50, 'dg-box', ['Clean image x₀']) + arrow('M130 61 H163') +
    box(165, 36, 150, 50, 'dg-amb', ['Add noise ε for a', 'random step t']) + arrow('M315 61 H348') +
    box(350, 36, 120, 50, 'dg-box', ['Noisy image x_t']) + arrow('M470 61 H503') +
    box(505, 36, 150, 50, 'dg-acc', ['Denoiser ε_θ(x_t, t)'], 'U-Net or transformer') + arrow('M655 61 H688') +
    box(690, 36, 120, 50, 'dg-red', ['Loss ‖ε − ε_θ‖²']) +
    arrow('M240 86 V110 H750 V88', 'dash') + label(495, 126, 'the noise that was added is the regression target') +
    label(44, 170, 'SAMPLING', 'dg-s dg-b') +
    box(10, 182, 120, 50, 'dg-box', ['Pure noise x_T']) + arrow('M130 207 H163') +
    box(165, 182, 130, 50, 'dg-acc', ['Denoiser'], 'predicts the noise') + arrow('M295 207 H328') +
    box(330, 182, 140, 50, 'dg-box', ['Remove predicted', 'noise (one step)']) + arrow('M470 207 H503') +
    box(505, 182, 90, 50, 'dg-box', ['x_{t−1}']) +
    arrow('M550 182 C 550 156, 230 156, 230 180') + label(390, 152, 'repeat for 20–1000 steps') +
    arrow('M595 207 H688') + label(642, 198, 'last step') +
    box(690, 182, 120, 50, 'dg-grn', ['New image x₀']) +
    box(165, 262, 220, 46, 'dg-pur', ['Prompt → text encoder']) +
    arrow('M230 262 V234', 'dash') + label(560, 290, 'steers the denoiser through cross-attention'),
    'Diffusion models. Training (top) is plain regression: noise a clean example to a random level and teach the network to predict that noise. Sampling (bottom) starts from pure noise and removes a little predicted noise per step. A text prompt conditions the denoiser through cross-attention; classifier-free guidance strengthens it.');

  // 5. U-Net
  diagram('unet', '0 0 820 290', 'Diagram of a U-Net encoder-decoder with skip connections',
    label(105, 22, 'input image') + label(715, 22, 'per-pixel output') +
    box(40, 30, 130, 44, 'dg-acc', ['full resolution']) + box(650, 30, 130, 44, 'dg-grn', ['full resolution']) +
    box(110, 90, 110, 44, 'dg-acc', ['½ resolution']) + box(600, 90, 110, 44, 'dg-grn', ['½ resolution']) +
    box(170, 150, 90, 44, 'dg-acc', ['¼']) + box(560, 150, 90, 44, 'dg-grn', ['¼']) +
    box(320, 220, 180, 48, 'dg-pur', ['bottleneck'], 'lowest resolution, most channels') +
    arrow('M105 74 L160 89') + arrow('M165 134 L210 149') + arrow('M215 194 L318 236') +
    arrow('M500 236 L603 195') + arrow('M605 150 L650 135') + arrow('M655 90 L710 75') +
    arrow('M170 52 H648', 'dash') + arrow('M220 112 H598', 'dash') + arrow('M260 172 H558', 'dash') +
    label(410, 44, 'skip connections copy high-resolution encoder features across') +
    label(140, 232, 'ENCODER: downsample', 'dg-s dg-b') + label(680, 232, 'DECODER: upsample', 'dg-s dg-b') +
    label(140, 248, 'what is in the image') + label(680, 248, 'where it is, pixel by pixel'),
    'U-Net. The encoder repeatedly halves the resolution to work out what is in the image; the decoder upsamples back to full resolution to say where. Skip connections (dashed) hand fine detail across, which is why U-Nets produce sharp segmentation masks, and why they became the classic denoiser of diffusion models.');

  // 6. The ML lifecycle
  diagram('lifecycle', '0 0 820 200', 'Diagram of the machine learning lifecycle',
    box(10, 50, 115, 60, 'dg-box', ['Frame the', 'problem'], 'goal & metric') + arrow('M125 80 H145') +
    box(147, 50, 115, 60, 'dg-amb', ['Data &', 'features'], 'collect · clean') + arrow('M262 80 H282') +
    box(284, 50, 115, 60, 'dg-acc', ['Train'], 'pipelines · tuning') + arrow('M399 80 H419') +
    box(421, 50, 115, 60, 'dg-acc', ['Evaluate'], 'test set · slices') + arrow('M536 80 H556') +
    box(558, 50, 115, 60, 'dg-grn', ['Deploy'], 'API or batch') + arrow('M673 80 H693') +
    box(695, 50, 115, 60, 'dg-pur', ['Monitor'], 'drift · quality') +
    arrow('M752 110 V150 H67 V112', 'acc') +
    label(410, 172, 'retrain or redesign when monitoring finds drift, new failure cases or a changed goal') +
    label(410, 28, 'version everything along the way: code, data, configuration, environment and model artifacts'),
    'The machine-learning lifecycle is a loop, not a line: what you learn in monitoring (drifting inputs, falling accuracy, new failure cases) feeds the next round of data work and training.');

  // 7. Cross-validation folds
  (function () {
    let rows = '';
    for (let i = 0; i < 5; i++) {
      const y = 80 + i * 34;
      for (let j = 0; j < 5; j++) {
        const x = 10 + j * 130;
        const val = i === j;
        rows += '<rect x="' + x + '" y="' + y + '" width="126" height="26" rx="5" class="' + (val ? 'dg-amb' : 'dg-acc') + '"/>' + label(x + 63, y + 17, val ? 'validate' : 'train', 'dg-s');
      }
      rows += label(738, y + 17, 'fold ' + (i + 1) + ' → score ' + (i + 1), 'dg-t');
    }
    diagram('cv', '0 0 820 280', 'Diagram of 5-fold cross-validation with a held-out test set',
      '<rect x="10" y="20" width="646" height="34" rx="6" class="dg-box"/>' + label(333, 42, 'labelled data for training and model selection', 'dg-t') +
      '<rect x="664" y="20" width="146" height="34" rx="6" class="dg-red"/>' + label(737, 42, 'test set', 'dg-t') +
      label(737, 70, 'locked until the end') +
      rows + label(410, 268, 'CV estimate = mean of the 5 scores (± their spread). Choose with it; report the test score once, at the end.'),
      'Five-fold cross-validation. Every example is used for validation exactly once and for training four times. Preprocessing and tuning must happen inside each fold, and the final test set stays untouched until the very end.');
  })();

  // 8. LoRA
  diagram('lora', '0 0 820 270', 'Diagram of a LoRA adapter next to a frozen weight matrix',
    box(10, 105, 80, 50, 'dg-box', ['input x'], 'length d_in') +
    arrow('M90 122 L168 82') + arrow('M90 138 L163 182') +
    box(170, 30, 200, 90, 'dg-box', ['W (frozen)'], 'd_out × d_in · pretrained') +
    box(165, 160, 130, 50, 'dg-acc', ['A'], 'r × d_in · trained') + arrow('M295 185 H318') +
    box(320, 160, 130, 50, 'dg-acc', ['B'], 'd_out × r · zero init') +
    plus(520, 130) +
    arrow('M370 75 L507 122') + arrow('M450 185 L507 139') + label(494, 180, '× α/r') +
    arrow('M534 130 H568') + box(570, 105, 240, 50, 'dg-grn', ['h = W x + (α/r) · B A x']) +
    label(410, 240, 'Only A and B are trained: r · (d_in + d_out) numbers instead of d_out · d_in.') +
    label(410, 258, 'After training, merge for free inference: W ← W + (α/r) · B A.'),
    'LoRA keeps the pretrained weight W frozen and learns a low-rank update B·A beside it. Because B starts at zero, training begins exactly at the base model; the rank r controls how much the adapter can change.');
  // 9. From base model to assistant: the LLM training pipeline
  diagram('pipeline', '0 0 820 245', 'Diagram of the LLM training pipeline from random weights to an assistant',
    box(59, 14, 170, 60, 'dg-box', ['Raw text'], 'trillions of tokens') +
    box(254, 14, 170, 60, 'dg-box', ['Demonstrations'], '(instruction, response) pairs') +
    box(474, 14, 170, 60, 'dg-box', ['Comparisons'], 'A is better than B') +
    box(10, 104, 100, 64, 'dg-box', ['Random', 'weights']) + arrow('M110 136 H178') +
    box(180, 104, 120, 64, 'dg-acc', ['Base model'], 'continues text') + arrow('M300 136 H378') +
    box(380, 104, 140, 64, 'dg-acc', ['Instruction', 'model']) + arrow('M520 136 H598') +
    box(600, 104, 210, 64, 'dg-grn', ['Assistant'], 'the chat model you talk to') +
    arrow('M144 74 V126', 'dash') + arrow('M339 74 V126', 'dash') + arrow('M559 74 V126', 'dash') +
    label(144, 196, 'PRETRAINING', 'dg-s dg-b') + label(144, 212, 'predict the next token') + label(144, 228, '→ knowledge, language, skills') +
    label(339, 196, 'SUPERVISED FINE-TUNING', 'dg-s dg-b') + label(339, 212, 'imitate good answers') + label(339, 228, '→ format and behaviour') +
    label(559, 196, 'PREFERENCE TUNING', 'dg-s dg-b') + label(559, 212, 'RLHF or DPO') + label(559, 228, '→ helpful, honest, harmless'),
    'Each stage reuses everything the previous one learned. Pretraining on raw text costs by far the most and gives the model its knowledge; supervised fine-tuning teaches it to answer in the right format; preference tuning shapes which of its possible answers it prefers. Reasoning models add a further reinforcement-learning stage with checkable rewards.');

  // 10. The LSTM cell
  diagram('lstm', '0 0 820 300', 'Diagram of an LSTM cell with forget, input and output gates',
    box(10, 40, 100, 40, 'dg-box', ['c_{t−1}']) + arrow('M110 60 H236') + plus(250, 60).replace('>+<', '>×<') + arrow('M264 60 H401') + plus(415, 60) + arrow('M429 60 H718') +
    box(720, 40, 90, 40, 'dg-grn', ['c_t']) +
    label(250, 32, 'forget: keep f ⊙ c_{t−1}') + label(415, 32, 'write: add i ⊙ g') + label(575, 50, 'CELL STATE (long-term memory)', 'dg-s dg-b') +
    box(205, 180, 90, 40, 'dg-amb', ['σ'], 'forget gate f') + box(315, 180, 90, 40, 'dg-amb', ['σ'], 'input gate i') +
    box(425, 180, 90, 40, 'dg-pur', ['tanh'], 'candidate g') + box(575, 180, 90, 40, 'dg-amb', ['σ'], 'output gate o') +
    arrow('M250 180 V76') + arrow('M360 180 V130 H399') + arrow('M470 180 V130 H431') + plus(415, 130).replace('>+<', '>×<') + arrow('M415 116 V76') +
    arrow('M680 60 V96') + box(650, 96, 60, 28, 'dg-pur', ['tanh']) + arrow('M680 124 V134') + plus(680, 150).replace('>+<', '>×<') + arrow('M620 180 V150 H664') +
    arrow('M694 150 H765 V233') + box(720, 235, 90, 40, 'dg-acc', ['h_t']) +
    box(10, 235, 120, 40, 'dg-box', ['[h_{t−1}, x_t]']) + '<path d="M130 255 H620" class="dg-line"/>' +
    arrow('M250 255 V222') + arrow('M360 255 V222') + arrow('M470 255 V222') + arrow('M620 255 V222') +
    label(375, 288, 'every gate reads the previous hidden state and the new input') + label(652, 272, 'h_t = o ⊙ tanh(c_t)'),
    'An LSTM cell. The cell state (top line) is the long-term memory: the forget gate scales the old memory, the input gate adds new candidate values, and because the update is mostly addition, gradients can flow back through many steps. The output gate decides how much of the memory to expose as the hidden state h_t. Each gate is a sigmoid between 0 (closed) and 1 (open).');

  // 11. Mixture of experts
  diagram('moe', '0 0 820 290', 'Diagram of a mixture-of-experts layer routing a token to two of four experts',
    box(10, 110, 100, 60, 'dg-box', ['token x'], 'one vector') + arrow('M110 140 H148') +
    box(150, 110, 120, 60, 'dg-pur', ['Router'], 'scores each expert') +
    box(380, 20, 120, 40, 'dg-box', ['Expert 1'], 'not run') + box(380, 80, 120, 40, 'dg-acc', ['Expert 2'], 'runs') +
    box(380, 140, 120, 40, 'dg-box', ['Expert 3'], 'not run') + box(380, 200, 120, 40, 'dg-acc', ['Expert 4'], 'runs') +
    arrow('M270 132 L378 42', 'dash') + arrow('M270 138 L378 100', 'acc') + arrow('M270 144 L378 160', 'dash') + arrow('M270 150 L378 218', 'acc') +
    label(350, 141, 'g = 0.7', 'dg-t') + label(300, 206, 'g = 0.3', 'dg-t') +
    arrow('M500 100 L577 134') + arrow('M500 220 L578 147') + plus(590, 140) + arrow('M604 140 H648') +
    box(650, 110, 160, 60, 'dg-grn', ['output y'], '0.7·FFN₂(x) + 0.3·FFN₄(x)') +
    label(410, 270, 'Only the top-k experts (here k = 2 of 4) run for each token: total parameters grow with the number of experts, compute per token does not.'),
    'A mixture-of-experts layer replaces one feed-forward network with several experts and a small router. The router scores the experts for each token, only the best k run, and their outputs are mixed with the router’s gate weights. Different tokens pick different experts.');

  // 12. Parallelism for large-scale training
  diagram('parallel', '0 0 820 300', 'Diagram comparing data, tensor and pipeline parallelism',
    label(140, 24, 'DATA PARALLEL', 'dg-s dg-b') + label(410, 24, 'TENSOR PARALLEL', 'dg-s dg-b') + label(680, 24, 'PIPELINE PARALLEL', 'dg-s dg-b') +
    '<path d="M275 40 V250" class="dg-line dash"/><path d="M545 40 V250" class="dg-line dash"/>' +
    box(20, 44, 110, 76, 'dg-acc', ['GPU 0', 'full model'], 'batch slice 1') + box(150, 44, 110, 76, 'dg-acc', ['GPU 1', 'full model'], 'batch slice 2') +
    arrow('M75 120 V158') + arrow('M205 120 V158') + box(30, 160, 220, 44, 'dg-amb', ['all-reduce'], 'average the gradients') +
    box(420, 36, 40, 30, 'dg-box', ['x']) + arrow('M432 66 L414 88') + arrow('M448 66 L496 88') +
    box(370, 90, 80, 60, 'dg-acc', ['GPU 0'], 'W[:, :n/2]') + box(460, 90, 80, 60, 'dg-pur', ['GPU 1'], 'W[:, n/2:]') +
    arrow('M410 150 V173') + arrow('M500 150 V173') + box(370, 175, 170, 32, 'dg-box', ['combine the halves']) +
    box(580, 40, 200, 40, 'dg-acc', ['layers 1–8'], 'GPU 0') + arrow('M680 80 V103') +
    box(580, 105, 200, 40, 'dg-acc', ['layers 9–16'], 'GPU 1') + arrow('M680 145 V168') +
    box(580, 170, 200, 40, 'dg-acc', ['layers 17–24'], 'GPU 2') +
    label(140, 228, 'split the batch; every GPU') + label(140, 244, 'holds the whole model') +
    label(410, 228, 'split each matrix; GPUs talk') + label(410, 244, 'inside every layer (fast links)') +
    label(680, 228, 'split the layers; micro-batches') + label(680, 244, 'flow through, with idle bubbles') +
    label(410, 286, 'Large runs combine all three, plus sharding of the optimizer state (ZeRO / FSDP).'),
    'Three ways to spread training over many GPUs. Data parallelism copies the model and splits the batch; tensor parallelism splits each weight matrix, so it needs very fast connections; pipeline parallelism places consecutive layers on different GPUs and keeps them busy by streaming micro-batches.');
  // ---------- Pictures that replace a paragraph (session 10) ----------
  const cell = (x, y, w, h, cls, text, tcls) => '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="5" class="' + cls + '"/>' + label(x + w / 2, y + h / 2 + 5, text, tcls || 'dg-t');

  // 13. Rules versus learning
  diagram('rules-vs-learning', '0 0 820 240', 'Diagram comparing traditional programming with machine learning',
    label(410, 24, 'TRADITIONAL PROGRAMMING: PEOPLE WRITE THE RULES', 'dg-s dg-b') +
    box(10, 36, 180, 56, 'dg-box', ['Rules'], 'written by people') + plus(214, 64) + box(238, 36, 120, 56, 'dg-box', ['Data']) +
    arrow('M358 64 H386') + box(388, 36, 150, 56, 'dg-pur', ['Program']) + arrow('M538 64 H566') +
    box(568, 36, 242, 56, 'dg-grn', ['Answers'], 'the program follows the rules') +
    label(410, 128, 'MACHINE LEARNING: THE RULES ARE LEARNED FROM EXAMPLES', 'dg-s dg-b') +
    box(10, 140, 180, 56, 'dg-box', ['Data'], 'many examples') + plus(214, 168) + box(238, 140, 120, 56, 'dg-box', ['Answers'], 'labels') +
    arrow('M358 168 H386') + box(388, 140, 150, 56, 'dg-acc', ['Learning'], 'an algorithm') + arrow('M538 168 H566') +
    box(568, 140, 242, 56, 'dg-grn', ['Rules: a model'], 'then used on new data') +
    label(410, 228, 'Machine learning turns the arrow around: from examples of answers to the rules that produce them.'),
    'In traditional programming people write the rules and the computer applies them to data. In machine learning you supply examples together with their answers, and a learning algorithm finds rules (a model) that reproduce them and, if all goes well, generalize to new data.');

  // 14. Names and objects in Python
  diagram('aliasing', '0 0 820 256', 'Diagram of two Python names bound to one list, and a third name bound to a copy',
    label(60, 22, 'NAMES', 'dg-s dg-b') + label(375, 22, 'OBJECTS IN MEMORY', 'dg-s dg-b') + label(690, 22, 'CODE', 'dg-s dg-b') +
    box(20, 40, 80, 44, 'dg-box', ['a']) + box(20, 110, 80, 44, 'dg-box', ['b']) + box(20, 180, 80, 44, 'dg-box', ['c']) +
    box(250, 50, 250, 64, 'dg-acc', ['[1, 2, 3, 4]'], 'one list object') + box(250, 166, 250, 60, 'dg-grn', ['[1, 2, 3]'], 'a new list: the copy') +
    arrow('M100 62 L247 76') + arrow('M100 132 L247 96') + arrow('M100 202 H247') +
    box(570, 34, 240, 192, 'dg-box', ['a = [1, 2, 3]', 'b = a', 'c = a.copy()', 'b.append(4)', 'print(a)', 'print(c)'], 'a has 4; c does not') +
    label(410, 248, 'Assignment never copies: b = a gives the same list a second name.'),
    'Python names are labels attached to objects. b = a attaches a second label to the same list, so a change made through b shows up through a; a.copy() makes a new list, which later changes do not touch.');

  // 15. Splitting data
  diagram('data-splits', '0 0 820 228', 'Diagram of a random split and a time-based split into training, validation and test sets',
    label(410, 22, 'RANDOM SPLIT: WHEN ROWS ARE INDEPENDENT OF EACH OTHER', 'dg-s dg-b') +
    cell(10, 32, 560, 40, 'dg-acc', 'training set: fit the model (about 70%)') + cell(574, 32, 118, 40, 'dg-amb', 'validation') + cell(696, 32, 114, 40, 'dg-red', 'test') +
    label(410, 100, 'TIME-BASED SPLIT: FORECASTING, OR ANYTHING THAT CHANGES OVER TIME', 'dg-s dg-b') +
    cell(10, 110, 560, 40, 'dg-acc', 'training set: the past') + cell(574, 110, 118, 40, 'dg-amb', 'validation') + cell(696, 110, 114, 40, 'dg-red', 'test') +
    arrow('M10 166 H806') + label(410, 186, 'time: the model only ever sees the past, as it will in production') +
    label(410, 214, 'Fit every preprocessing step on the training set only, and use the test set once, at the end.'),
    'Random splits suit rows that are independent of each other. When the data changes over time, split by time instead, so that validation and testing measure what production will see: a model trained on the past, judged on the future. Rows from one person or one patient also belong together, in a single split.');

  // 16. One training step in PyTorch
  diagram('train-step', '0 0 820 214', 'Diagram of one training step in PyTorch: zero the gradients, forward pass, loss, backward pass, optimizer step',
    label(410, 40, 'clear old gradients first: backward() adds to .grad instead of replacing it') +
    box(10, 70, 100, 60, 'dg-box', ['batch'], 'x and y') + arrow('M110 100 H128') +
    box(130, 70, 125, 60, 'dg-box', ['zero_grad'], 'opt.zero_grad()') + arrow('M255 100 H273') +
    box(275, 70, 125, 60, 'dg-acc', ['forward'], 'pred = model(x)') + arrow('M400 100 H418') +
    box(420, 70, 125, 60, 'dg-amb', ['loss'], 'loss_fn(pred, y)') + arrow('M545 100 H563') +
    box(565, 70, 115, 60, 'dg-pur', ['backward'], 'loss.backward()') + arrow('M680 100 H698') +
    box(700, 70, 110, 60, 'dg-grn', ['step'], 'opt.step()') +
    arrow('M755 130 V164 H60 V132', 'acc') + label(410, 192, 'repeat for every batch; one pass over all the batches is an epoch'),
    'Every training loop in PyTorch repeats the same five moves for each batch: clear the old gradients, run the model forward, compute the loss, backpropagate, and let the optimizer update the weights. Switch the model to model.eval() and wrap evaluation in torch.no_grad() when you are not training.');

  // 17. Bag of words
  (function () {
    const words = ['the', 'cat', 'dog', 'sat', 'ran'], docs = [['the cat sat', [1, 1, 0, 1, 0]], ['the dog sat', [1, 0, 1, 1, 0]], ['the cat ran', [1, 1, 0, 0, 1]]];
    let grid = '';
    words.forEach((w, j) => { grid += label(356 + j * 92, 40, w, 'dg-t'); });
    docs.forEach(([text, counts], i) => {
      const y = 50 + i * 50;
      grid += cell(10, y, 210, 40, 'dg-box', 'd' + (i + 1) + ': ' + text);
      counts.forEach((n, j) => { grid += cell(310 + j * 92, y, 88, 40, n ? 'dg-acc' : 'dg-box', String(n)); });
    });
    diagram('bow', '0 0 820 240', 'Diagram of three short documents turned into a document-term matrix of word counts',
      grid + arrow('M224 120 H304') + label(410, 212, 'Each document becomes a row of word counts. Word order is lost, but rows can be compared,') +
      label(410, 230, 'and TF-IDF reweights the columns so that words found in every document count for less.'),
      'Bag-of-words turns each document into a vector with one entry per word of the vocabulary. The matrix is mostly zeros for real text, so libraries store it sparsely; any classifier can then learn from the rows.');
  })();

  // 18. Four families of generative models
  diagram('gen-families', '0 0 820 290', 'Diagram comparing autoencoders, variational autoencoders, GANs and diffusion models',
    box(10, 20, 120, 50, 'dg-pur', ['Autoencoder']) + box(150, 20, 150, 50, 'dg-box', ['encoder'], 'x to a short code z') + arrow('M300 45 H328') +
    box(330, 20, 150, 50, 'dg-acc', ['decoder'], 'z back to x') + arrow('M480 45 H508') + box(510, 20, 300, 50, 'dg-grn', ['reconstruction'], 'learns to compress; not a generator') +
    box(10, 90, 120, 50, 'dg-pur', ['VAE']) + box(150, 90, 150, 50, 'dg-box', ['encoder'], 'x to a mean and a spread') + arrow('M300 115 H328') +
    box(330, 90, 150, 50, 'dg-acc', ['sample z'], 'a smooth latent space') + arrow('M480 115 H508') + box(510, 90, 300, 50, 'dg-grn', ['decoder: new samples'], 'from a random z') +
    box(10, 160, 120, 50, 'dg-pur', ['GAN']) + box(150, 160, 150, 50, 'dg-box', ['generator'], 'noise to a fake x') + arrow('M300 185 H328') +
    box(330, 160, 150, 50, 'dg-acc', ['discriminator'], 'real or fake?') + arrow('M480 185 H508') + box(510, 160, 300, 50, 'dg-grn', ['two networks compete'], 'sharp samples; unstable training') +
    box(10, 230, 120, 50, 'dg-pur', ['Diffusion']) + box(150, 230, 150, 50, 'dg-box', ['add noise'], 'data to pure noise') + arrow('M300 255 H328') +
    box(330, 230, 150, 50, 'dg-acc', ['learn to denoise'], 'predict the noise') + arrow('M480 255 H508') + box(510, 230, 300, 50, 'dg-grn', ['sample step by step'], 'from pure noise to data'),
    'Four ways to learn to generate. An autoencoder compresses and rebuilds; a VAE makes its code space smooth enough to sample from; a GAN trains a generator against a critic; a diffusion model learns to undo noise one small step at a time, and generates by starting from pure noise.');

  // 19. Vision transformer
  diagram('vit', '0 0 820 214', 'Diagram of a vision transformer turning an image into patch tokens',
    label(410, 40, 'An image becomes a sequence: each 16 × 16 patch is one token, like a word in a sentence.') +
    box(10, 80, 120, 70, 'dg-box', ['image'], '224 × 224 pixels') + arrow('M130 115 H158') +
    box(160, 80, 130, 70, 'dg-amb', ['196 patches'], '16 × 16 each') + arrow('M290 115 H318') +
    box(320, 80, 145, 70, 'dg-acc', ['linear embedding'], '+ position') + arrow('M465 115 H488') +
    box(490, 80, 155, 70, 'dg-pur', ['transformer', 'encoder'], 'CLS + 196 tokens') + arrow('M645 115 H668') +
    box(670, 80, 140, 70, 'dg-grn', ['class'], 'from the CLS token') +
    label(410, 196, 'Without convolution, ViTs learn locality from data, so they need more data or pretraining than CNNs.'),
    'A vision transformer cuts the image into patches, flattens each one, projects it to a vector and adds a position embedding. A learned CLS token joins the sequence, and after the transformer encoder its vector feeds the classifier.');

  // 20. Split, apply, combine
  (function () {
    const rows = [['A', 10], ['B', 5], ['A', 20], ['C', 7], ['B', 15], ['C', 3]];
    let t = label(92, 26, 'the DataFrame', 'dg-s dg-b') + label(330, 26, 'split by city', 'dg-s dg-b') + label(520, 26, 'apply sum()', 'dg-s dg-b') + label(724, 26, 'combine', 'dg-s dg-b');
    rows.forEach(([c, v], i) => { const y = 40 + i * 32; t += cell(20, y, 70, 28, 'dg-box', c) + cell(94, y, 80, 28, 'dg-box', String(v)); });
    const groups = [['A: 10, 20', 30, 'dg-acc'], ['B: 5, 15', 20, 'dg-amb'], ['C: 7, 3', 10, 'dg-pur']];
    groups.forEach(([g, s, cls], i) => {
      const y = 40 + i * 70;
      t += box(260, y, 140, 56, cls, [g]) + arrow('M178 136 L256 ' + (y + 28)) + box(460, y, 120, 56, cls, ['sum: ' + s]) + arrow('M400 ' + (y + 28) + ' H456');
      t += cell(660, 80 + i * 34, 60, 28, 'dg-grn', g[0]) + cell(724, 80 + i * 34, 80, 28, 'dg-grn', String(s)) + arrow('M580 ' + (y + 28) + ' L656 ' + (94 + i * 34));
    });
    diagram('groupby', '0 0 820 262', 'Diagram of split-apply-combine: grouping rows by city and summing sales',
      t + label(410, 254, 'df.groupby("city")["sales"].sum(): split the rows by key, apply a function to each group, combine the results.'),
      'groupby splits the rows into groups by the key column, applies a function such as sum or mean to each group, and combines the answers into a new table with one row per key. agg applies several functions at once; transform returns a value for every original row.');
  })();

  // 21. Choosing a chart
  diagram('chart-chooser', '0 0 820 256', 'Diagram mapping six questions to the charts that answer them',
    box(10, 30, 260, 90, 'dg-acc', ['How is one variable', 'distributed?'], 'histogram, box plot') +
    box(280, 30, 260, 90, 'dg-acc', ['How do two numbers', 'relate?'], 'scatter plot') +
    box(550, 30, 260, 90, 'dg-acc', ['How does it change', 'over time?'], 'line chart') +
    box(10, 140, 260, 90, 'dg-grn', ['How do categories', 'compare?'], 'bar chart, sorted') +
    box(280, 140, 260, 90, 'dg-grn', ['What are the parts', 'of a whole?'], 'stacked bar (rarely a pie)') +
    box(550, 140, 260, 90, 'dg-grn', ['How do groups differ', 'in distribution?'], 'side-by-side box plots, facets') +
    label(410, 248, 'Start from the question the reader has, then pick the chart that answers it most directly.'),
    'The chart follows from the question: distributions call for histograms and box plots, relationships for scatter plots, change over time for line charts, comparisons between categories for sorted bars, and parts of a whole for stacked bars.');

  // 22. Sequence to sequence with attention
  diagram('seq2seq', '0 0 820 300', 'Diagram of an encoder-decoder translating a sentence, with attention from a decoder step to every encoder state',
    box(20, 230, 100, 40, 'dg-box', ['le']) + box(140, 230, 100, 40, 'dg-box', ['chat']) + box(260, 230, 100, 40, 'dg-box', ['noir']) +
    arrow('M70 230 V202') + arrow('M190 230 V202') + arrow('M310 230 V202') +
    box(20, 150, 100, 50, 'dg-acc', ['h1']) + arrow('M120 175 H138') + box(140, 150, 100, 50, 'dg-acc', ['h2']) + arrow('M240 175 H258') + box(260, 150, 100, 50, 'dg-acc', ['h3']) +
    box(360, 50, 100, 50, 'dg-amb', ['context'], 'weighted sum') +
    arrow('M70 150 L370 102') + arrow('M190 150 L392 102') + arrow('M310 150 L420 102') + label(214, 120, '0.1') + label(282, 128, '0.2') + label(352, 136, '0.7') +
    arrow('M360 175 H478') + label(419, 168, 'start') +
    box(480, 150, 90, 50, 'dg-pur', ['s1']) + arrow('M570 175 H598') + box(600, 150, 90, 50, 'dg-pur', ['s2']) + arrow('M690 175 H718') + box(720, 150, 90, 50, 'dg-pur', ['s3']) +
    arrow('M460 82 L610 148') +
    arrow('M525 150 V92') + arrow('M645 150 V92') + arrow('M765 150 V92') +
    box(480, 40, 90, 50, 'dg-grn', ['the']) + box(600, 40, 90, 50, 'dg-grn', ['black']) + box(720, 40, 90, 50, 'dg-grn', ['cat']) +
    label(190, 292, 'encoder: reads the whole input') + label(645, 226, 'decoder: writes one word per step'),
    'Translating "le chat noir". The encoder turns the input into one state per word. At each output step the decoder scores every encoder state against its own state, and the softmax of those scores weights a context vector: here step 2 attends mostly to "noir" (0.7) to produce "black". Without attention, the whole sentence had to squeeze through the final state alone.');

  // 23. Attention masks: bidirectional and causal
  (function () {
    const toks = ['the', 'cat', '[MASK]', 'on', 'mat'];
    const grid = (x0, causal) => {
      let s = '';
      toks.forEach((t, j) => { s += label(x0 + 20 + j * 40, 44, t, 'dg-s'); });
      toks.forEach((t, i) => {
        s += label(x0 - 36, 50 + i * 30 + 20, t, 'dg-s');
        for (let j = 0; j < 5; j++) s += '<rect x="' + (x0 + j * 40) + '" y="' + (50 + i * 30) + '" width="38" height="28" rx="4" class="' + (!causal || j <= i ? 'dg-acc' : 'dg-box') + '"/>';
      });
      return s;
    };
    diagram('attn-masks', '0 0 820 256', 'Diagram of a bidirectional attention mask next to a causal attention mask',
      label(190, 22, 'BIDIRECTIONAL (BERT): EVERY TOKEN SEES EVERY TOKEN', 'dg-s dg-b') + label(620, 22, 'CAUSAL (GPT): EACH TOKEN SEES ONLY THE PAST', 'dg-s dg-b') +
      grid(90, false) + grid(520, true) +
      label(190, 224, 'BERT fills in [MASK] using both sides') + label(620, 224, 'GPT predicts the next token from the left') +
      label(410, 246, 'rows: the token that is looking; filled squares: the tokens it may attend to'),
      'Each row is a token, and its filled squares are the positions it may attend to. An encoder like BERT uses the full square, so a masked word is predicted from both sides. A decoder like GPT uses the lower triangle, so no token can see the future it is trained to predict.');
  })();

  // 24. Joins that duplicate or drop rows
  (function () {
    let t = label(72, 46, 'orders', 'dg-s dg-b') + label(337, 46, 'users', 'dg-s dg-b') + label(664, 46, 'orders LEFT JOIN users', 'dg-s dg-b');
    [['o1', 'u1'], ['o2', 'u2'], ['o3', 'u3']].forEach(([o, u], i) => { t += cell(10, 60 + i * 32, 60, 28, 'dg-box', o) + cell(74, 60 + i * 32, 60, 28, 'dg-box', u); });
    [['u1', 'basic'], ['u2', 'pro'], ['u2', 'pro trial']].forEach(([u, p], i) => { t += cell(250, 60 + i * 32, 60, 28, i === 2 ? 'dg-red' : 'dg-box', u) + cell(314, 60 + i * 32, 110, 28, i === 2 ? 'dg-red' : 'dg-box', p); });
    [['o1', 'u1', 'basic', 'dg-acc'], ['o2', 'u2', 'pro', 'dg-acc'], ['o2', 'u2', 'pro trial', 'dg-red'], ['o3', 'u3', 'NULL', 'dg-amb']].forEach(([o, u, p, cls], i) => {
      t += cell(520, 60 + i * 32, 60, 28, cls, o) + cell(584, 60 + i * 32, 60, 28, cls, u) + cell(648, 60 + i * 32, 160, 28, cls, p);
    });
    diagram('joins', '0 0 820 250', 'Diagram of a left join where a duplicated key repeats an order and a missing key gives NULL',
      t + arrow('M428 108 H514') + label(472, 96, 'on user_id') +
      label(410, 214, 'A duplicated key multiplies rows: order o2 now appears twice, so its revenue would be counted twice.') +
      label(410, 236, 'A missing key gives NULL (LEFT JOIN) or drops the row (INNER JOIN). Count rows before and after every join.'),
      'Joins go wrong in two quiet ways. If the right table repeats a key, every matching row on the left is repeated; if a key is missing, a left join fills in NULL while an inner join silently drops the row. Check the row count and the key uniqueness of both tables before you join.');
  })();

  // 25. The Model Context Protocol
  diagram('mcp', '0 0 820 280', 'Diagram of an MCP host with two clients connected to two servers',
    '<rect x="10" y="30" width="250" height="230" rx="10" class="dg-box"/>' + label(135, 56, 'Host application', 'dg-t') + label(135, 74, 'a chat app or an IDE') +
    box(30, 92, 210, 44, 'dg-acc', ['MCP client 1']) + box(30, 150, 210, 44, 'dg-acc', ['MCP client 2']) + box(30, 208, 210, 40, 'dg-pur', ['the language model']) +
    box(560, 40, 250, 90, 'dg-grn', ['MCP server: files'], 'tools · resources · prompts') +
    box(560, 160, 250, 90, 'dg-grn', ['MCP server: a database'], 'tools · resources · prompts') +
    arrow('M240 114 L556 86') + label(400, 86, 'stdio: a local process') +
    arrow('M240 172 L556 204') + label(400, 206, 'Streamable HTTP: remote') +
    label(410, 274, 'one client per server; the host decides what the model may call'),
    'The host (a chat app or an IDE) runs one MCP client for each server it connects to. Servers offer tools the model can call, resources it can read and prompts the user can pick, over JSON-RPC: stdio for a local process, Streamable HTTP for a remote one. The host stays in charge and asks the user before risky actions.');

  // 26. Equivariance
  diagram('equivariance', '0 0 820 262', 'Diagram showing that an equivariant function commutes with rotation',
    box(60, 30, 200, 60, 'dg-box', ['input x']) + box(545, 30, 230, 60, 'dg-box', ['rotated input'], 'g · x') +
    box(60, 170, 200, 60, 'dg-acc', ['features f(x)']) + box(545, 170, 230, 60, 'dg-acc', ['f(g · x) = g · f(x)'], 'the features rotate too') +
    arrow('M260 60 H541') + label(400, 50, 'rotate by g') + arrow('M260 200 H541') + label(400, 190, 'rotate by g') +
    arrow('M160 90 V166') + label(208, 134, 'apply f') + arrow('M660 90 V166') + label(708, 134, 'apply f') +
    label(410, 254, 'Invariant: f(g · x) = f(x), the output does not change. Equivariant: the output moves with the input.'),
    'Going down then across or across then down gives the same result: that is what equivariance means. Convolutions are equivariant to shifts, and equivariant networks build in rotations or permutations the same way, so they need not learn each orientation from data.');
})(typeof globalThis !== 'undefined' ? globalThis : window);
