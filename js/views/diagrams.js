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
})(typeof globalThis !== 'undefined' ? globalThis : window);
