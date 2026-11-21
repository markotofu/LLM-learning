/* ML Atlas – content: video models, world models, and robot learning with vision-language-action models */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'video-generation', branch: 'spec', title: 'Video Models: Understanding and Generation', short: 'Video Models', level: 3, minutes: 50, prereqs: ['diffusion-models', 'vision-transformers'],
    tags: ['video', 'video generation', 'text-to-video', 'image-to-video', 'video understanding', 'action recognition', 'latent diffusion', 'diffusion transformer', 'spacetime patches', 'temporal attention', 'video vae', 'fvd', 'deepfakes', 'provenance', 'c2pa', 'watermark'],
    summary: 'How models understand and generate video: action recognition with 3D convolutions and video transformers, video language models, latent video diffusion with spacetime patches, keeping frames consistent and extending clips, the enormous token budgets involved, evaluating generated video, and the risks of realistic synthetic footage.',
    why: `A video is a stack of images that must agree with each other: the same objects, lighting and physics from one frame to the next. Video models face two problems that images do not: **time**, because motion and consistency matter as much as the content of any single frame, and **size**, because a few seconds of video hold as many numbers as hundreds of images. How they compress and attend across space and time explains what today's video generators can do, why they are so expensive, and where they still fail.`,
    objectives: [
      'Describe how video is recognized and understood: 3D convolutions, video transformers and video language models',
      'Explain latent video diffusion: a video VAE, spacetime patches and a diffusion transformer',
      'Estimate the token count and attention cost of a clip, and explain why compression and factorized attention matter',
      'Explain how models keep frames consistent, why extending a clip frame by frame drifts, and how generated video is evaluated and labelled'
    ],
    sections: [
      { type: 'text', title: "From images to video", md: `A video is a sequence of frames: T images of H × W pixels with 3 colour channels each. Five seconds of 720p video at 24 frames per second is 120 frames × 1,280 × 720 pixels × 3 ≈ **332 million numbers**, about 420 times as many as one 512 × 512 image. Two things make video harder than images:

- **Time.** Neighbouring frames are nearly identical, but not quite: objects move, the camera pans, the light changes. A model has to capture motion and keep everything consistent across the whole clip: the same face, the same cup, and physics that makes sense.
- **Size.** Everything that was affordable for one image is now repeated over a hundred frames, and attention, which compares every token with every other, grows with the square of the number of tokens.

The same building blocks serve two families of tasks:

| Understanding: video in, labels or text out | Generation: text or an image in, video out |
| --- | --- |
| action recognition ("someone is juggling"), finding when an event happens, captioning, video question answering, search | text-to-video, image-to-video (animating a still), video-to-video (restyling or editing), extending a clip, frame interpolation |` },
      { type: 'text', title: "Understanding video: from 3D convolutions to video language models", md: `- **3D convolutions** extend a CNN's filters across time, for example 3 × 3 × 3 over frames, height and width, so that a filter can respond to motion as well as to shape (see [Convolutional Neural Networks](#/concept/cnn)). I3D (2017) "inflated" the 2D filters of an ImageNet-trained network into 3D ones, so that video models could reuse image training.
- **Two-stream networks** process the raw frames and the *optical flow* (how each pixel moves between frames) in separate networks and combine their predictions.
- **Video transformers** cut the clip into spacetime patches, like the patches of a vision transformer but spanning a few frames, and attend across them (see [Vision Transformers](#/concept/vision-transformers)). To save compute, TimeSformer (2021) alternates attention within each frame with attention across frames at the same position: *divided*, or factorized, attention.
- **Video language models** sample a handful of frames (often one or two per second), encode each with an image encoder and pass the frame tokens to a language model together with the question (see [Multimodal Models](#/concept/multimodal-models)). Any video task then becomes question answering, at the price of missing fast events that fall between the sampled frames.

Results depend heavily on how frames are sampled: a 30-second clip at 30 frames per second has 900 frames, and models typically see 8 to 64 of them.` },
      { type: 'text', title: "Generating video: latent diffusion over spacetime patches", md: `Today's video generators, including OpenAI's Sora, Google DeepMind's Veo and open-weight models such as Wan, HunyuanVideo and CogVideoX, share one recipe, a direct extension of latent diffusion for images (see [Diffusion Models & Flow Matching](#/concept/diffusion-models)):

1. **Compress.** A video VAE (variational autoencoder) encodes the clip into a much smaller latent. A typical design compresses 4× in time and 8× in each spatial direction, keeping 16 channels per latent position.
2. **Cut into patches.** The latent is divided into spacetime patches, for example 2 × 2 latent positions of one latent frame, and each patch becomes a token.
3. **Denoise.** A diffusion transformer (DiT) starts from noise and removes it step by step, attending across all the tokens. Many recent models are trained with flow matching. The prompt enters through cross-attention or by being processed together with the video tokens, and classifier-free guidance strengthens its influence.
4. **Decode.** The VAE decoder turns the clean latent back into frames.

The number of tokens the transformer has to handle is

$$ N = ⌈T / c_t⌉ · (H / (c_s · p)) · (W / (c_s · p)) $$
:::where
N — the number of tokens (spacetime patches) the transformer processes
T, H, W — the number of frames, and the height and width of each frame in pixels
c_t — the VAE's compression in time (often 4)
c_s — the VAE's compression in each spatial direction (often 8)
p — the patch size in latent positions (often 2)
⌈ ⌉ — rounding up to a whole number
:::

and full attention costs grow with N², so a longer or sharper clip is far more expensive than it looks.` },
      { type: 'code', title: "How many tokens is a video clip?", code: `import math

def video_tokens(frames, height, width, t_comp=4, s_comp=8, patch=2):
    """Tokens a diffusion transformer sees: compress with a video VAE, then cut into patches."""
    latent_frames = math.ceil(frames / t_comp)
    return latent_frames * (height // (s_comp * patch)) * (width // (s_comp * patch))

image = video_tokens(1, 512, 512)
print(f"one 512x512 image: {image:,} tokens")
print("clip                      pixel values    tokens   vs image   attention cost vs image")
for name, seconds, h, w in [("2 s at 480p", 2, 480, 848), ("5 s at 720p", 5, 720, 1280),        # 848 and 1088: sizes
                            ("10 s at 720p", 10, 720, 1280), ("10 s at 1080p", 10, 1088, 1920)]:  # divisible by 16
    frames = 24 * seconds
    n = video_tokens(frames, h, w)
    print(f"{name:22s} {frames * h * w * 3:15,} {n:9,} {n / image:9.0f}x {(n / image) ** 2:16,.0f}x")
five, ten = video_tokens(120, 720, 1280), video_tokens(240, 720, 1280)
print("doubling the length quadruples the attention cost:", (ten / five) ** 2 == 4)`, after: `A 5-second 720p clip holds about a hundred images' worth of tokens, but because attention compares every token with every other, full attention over it costs about 11,000 times as much as for one image. Doubling the length doubles the tokens and quadruples the attention cost. The VAE's 256-fold compression (4 × 8 × 8) is what makes generation feasible at all, and it is why video models are trained on short, low-resolution clips first and on longer, sharper ones later.`, predict: {"type":"mcq","q":"Doubling a clip's length doubles its number of tokens. How much more does full attention over the clip cost?","options":["The same","2×","4×","8×"],"answer":2,"explain":"Full attention compares every pair of tokens, so its cost grows with N²: twice the tokens means four times the pairs.","expect":["doubling the length quadruples the attention cost: True"]} },
      { type: 'chart', title: "Tokens in a clip as it gets longer", md: `The token formula above with a 4 × 8 × 8 video VAE and 2 × 2 patches:`, after: `Tokens grow in a straight line with length and in proportion to the number of pixels per frame, so a 20-second 1080p clip has close to a million tokens, about a thousand images' worth. Full attention cost grows with the square of these numbers, which is why most models generate short clips and extend them.`, chart: {"kind":"line","x":[1,2,3,4,5,6,8,10,12,15,20],"yLog":true,"series":[{"name":"480p at 24 frames per second","y":[9540,19080,28620,38160,47700,57240,76320,95400,114480,143100,190800]},{"name":"720p at 24 frames per second","y":[21600,43200,64800,86400,108000,129600,172800,216000,259200,324000,432000]},{"name":"1080p at 24 frames per second","y":[48960,97920,146880,195840,244800,293760,391680,489600,587520,734400,979200]}],"xLabel":"clip length (seconds)","yLabel":"tokens (log scale)","hlines":[{"y":1024,"label":"one 512 × 512 image"}]} },
      { type: 'check', question: {"type":"mcq","q":"A video VAE compresses 4× in time and 8× in each spatial direction. How many pixel positions does each latent position stand for?","options":["32","64","256","512"],"answer":2,"explain":"4 (time) × 8 (height) × 8 (width) = 256 pixel positions per latent position; patches of 2 × 2 latent positions then make each token stand for 1,024 of them."} },
      { type: 'text', title: "Keeping frames consistent", md: `Generating all the frames **together**, as one spacetime latent, lets every patch attend to every other, so that an object is described consistently throughout the clip. Earlier systems generated frames one at a time or kept space and time largely apart, and their videos flickered and morphed.

Attention over the whole clip is expensive, so some models **factorize** it: attend within each frame, then across time at each spatial position.

$$ full: (T · S)²     factorized: T · S² + S · T² $$
:::where
T — the number of latent frames
S — the number of tokens in one latent frame
(T · S)² — the pairs of tokens compared by full spacetime attention
T · S² + S · T² — the pairs compared by attention within each frame plus attention across time at each position
:::

For a 5-second 720p clip that is about 30 times fewer pairs, but information between distant frames then travels only indirectly, which weakens long-range consistency, and many recent models pay for full attention.

Two more techniques matter in practice:

- **Going beyond the training length.** Models are trained on clips of a few seconds. Longer videos are made by *extension*, generating the next chunk conditioned on the last frames of the previous one, or by generating keyframes and filling the gaps. Each extension step adds small errors, so identities, colours and motion slowly **drift**.
- **Conditioning beyond text.** Image-to-video fixes the first frame. Other controls include camera paths, motion strokes, depth maps and reference images of a character.` },
      { type: 'code', title: "Why frames must be generated together", md: `A toy version of the problem: each "frame" is a row of 96 pixels with one bright blob, and the prompt asks for the blob to move right by one pixel per frame. Three ways to make a 48-frame clip:`, code: `import numpy as np
rng = np.random.default_rng(0)
P, T, speed = 96, 48, 1.0                  # pixels per frame, frames, the motion the prompt asks for (pixels per frame)
grid = np.arange(P)

def render(positions):                     # each frame: a bright blob at the given position
    return np.exp(-0.5 * ((grid[None, :] - np.asarray(positions)[:, None]) / 3.0) ** 2)

def generate(strategy, start=20.0):
    if strategy == "each frame on its own":           # a perfect image model, but no link between frames
        pos = rng.uniform(10, P - 10, T)
    elif strategy == "frame by frame":                # each frame conditioned on the previous one, small errors
        pos = start + np.cumsum(np.r_[0.0, speed + rng.normal(0, 0.3, T - 1)])
    else:                                             # all frames generated together, as one spacetime latent
        pos = start + speed * np.arange(T)
    return pos, render(pos)

target = 20.0 + speed * np.arange(T)
print("strategy                 flicker   position error at the last frame")
results = {}
for strategy in ("each frame on its own", "frame by frame", "all frames together"):
    flicker, drift = [], []
    for _ in range(200):
        pos, frames = generate(strategy)
        flicker.append(np.abs(np.diff(frames, axis=0)).mean())
        drift.append(abs(pos[-1] - target[-1]))
    results[strategy] = (np.mean(flicker), np.mean(drift))
    print(f"{strategy:22s} {np.mean(flicker):9.3f} {np.mean(drift):12.1f} pixels")
print("independent frames flicker far more:", bool(results["each frame on its own"][0] > 5 * results["all frames together"][0]))
print("frame by frame drifts, joint generation does not:", bool(results["frame by frame"][1] > 1 and results["all frames together"][1] < 0.01))`, after: `Each frame of the first strategy is a perfect image of a blob, yet the video flickers wildly, almost seven times more than the others: nothing ties one frame to the next. Generating frame by frame, each frame conditioned on the previous one, is smooth, but the small error of each step adds up: after 48 frames the blob is on average about 1.7 pixels from where the prompt's motion puts it, and the gap keeps growing with length. Generating all frames together keeps the motion both smooth and on track. Real models face the same trade-off with identities, colours and physics instead of one blob's position.`, predict: {"type":"mcq","q":"A perfect image model draws each frame of a clip independently, from the same prompt. What will the video look like?","options":["Smooth and consistent, because every frame is perfect","Flickering: each frame is fine on its own, but consecutive frames disagree","Blurry, because the frames are averaged","The same as generating frame by frame"],"answer":1,"explain":"Independent samples share nothing but the prompt, so the blob jumps to a new place in every frame. Consistency comes from modelling the frames jointly.","expect":["independent frames flicker far more: True","frame by frame drifts, joint generation does not: True"]} },
      { type: 'chart', title: "How frame-by-frame generation drifts", md: `The blob example extended to ten seconds: the average distance between where frame-by-frame generation puts the blob and where the prompt's motion says it should be:`, after: `The error grows like the square root of the number of frames, the signature of a random walk: small, independent mistakes that are never corrected. Doubling the per-frame error doubles the drift at every length. Joint generation has no such accumulation within the clip it generates, which is why models generate as long a clip as they can afford in one go.`, chart: {"kind":"line","x":[0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240],"points":false,"series":[{"name":"frame by frame, small error per frame (σ = 0.3 px)","y":[0,0.677,0.957,1.173,1.354,1.514,1.658,1.791,1.915,2.031,2.141,2.245,2.345,2.441,2.533,2.622,2.708,2.791,2.872,2.951,3.028,3.103,3.176,3.247,3.317,3.385,3.452,3.518,3.582,3.646,3.708]},{"name":"frame by frame, larger error per frame (σ = 0.6 px)","y":[0,1.354,1.915,2.345,2.708,3.028,3.317,3.582,3.83,4.062,4.282,4.491,4.691,4.882,5.066,5.244,5.416,5.583,5.745,5.902,6.056,6.205,6.351,6.494,6.633,6.77,6.904,7.036,7.165,7.292,7.416]},{"name":"all frames generated together","y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"dash":true}],"xLabel":"frame (240 frames = 10 seconds at 24 fps)","yLabel":"average position error (pixels)","yMin":0} },
      { type: 'check', question: {"type":"mcq","q":"Why do long generated videos often drift, with a face slowly changing or colours shifting?","options":["The VAE is too small","They are built by extending clips chunk by chunk, and each step adds small errors","The model forgets the prompt after one second","Attention cannot handle more than ten frames"],"answer":1,"explain":"Each extension is conditioned on frames that already contain small errors, so the errors accumulate, much like the frame-by-frame strategy in the code above."} },
      { type: 'text', title: "Evaluating generated video", md: `- **Distribution metrics.** The Fréchet Video Distance (FVD) embeds real and generated clips with a pretrained video classifier and compares the two clouds of feature vectors, the video counterpart of FID for images. Lower is better. It needs many clips, and it depends on the classifier behind it, which is more sensitive to appearance than to motion.
- **Text alignment.** Image–text similarity models such as CLIP scored frame by frame, or a vision-language model asked questions about the clip ("is the dog jumping over the fence?").
- **Human preference.** Side-by-side comparisons, often aggregated into Elo-style leaderboards, remain the most trusted signal. Benchmark suites such as VBench split quality into separate dimensions: subject consistency, motion smoothness, aesthetic quality, how well the motion follows the prompt, and more.
- **Failure modes to test for:** objects that appear or vanish (object permanence), impossible physics (liquids, collisions, gravity), hands and on-screen text, identities that morph, and motion that freezes or loops in longer clips.` },
      { type: 'text', title: "Realistic video: risks and provenance", md: `Synthetic video that is indistinguishable from real footage creates serious risks:

- **Harms:** deepfakes of real people, non-consensual intimate imagery, fraud (a cloned face and voice on a video call), and fabricated "evidence" in news and politics.
- **Policies:** consent and likeness rules, refusing prompts that put real people or minors into harmful content, and extra checks for public figures.
- **Provenance:** C2PA *content credentials* attach signed metadata that records how a file was made, and invisible watermarks (such as Google DeepMind's SynthID) survive some editing and let a provider recognize its own outputs. Both help, and neither is proof: metadata can be stripped, watermarks can be weakened by re-encoding or cropping, and detectors that guess whether footage is synthetic make mistakes in both directions.
- **Cost and energy:** a few seconds of high-resolution video can take minutes of GPU time, which shapes prices, clip-length limits and the environmental footprint (see [Responsible AI](#/concept/responsible-ai)).` }
    ],
    when: `- Recognizing actions or events in footage (sports analytics, safety monitoring, content moderation): a video classifier or a video language model.
- Generating clips from text or a still image (marketing drafts, previsualization, game and film prototyping): a video diffusion model, usually through an API.
- Not for exact, controllable motion (product demos that must match reality, legal evidence): generated video is plausible, not faithful.`,
    pitfalls: `- Treating every frame as an independent image: the result flickers and objects change identity.
- Underestimating cost: tokens grow with length × resolution, and full attention grows with their square.
- Judging a model by a few cherry-picked clips: evaluate on many prompts, including physics and long-range consistency.
- Publishing realistic video of real people without consent, or without provenance labels.`,
    takeaways: [
      'Video models work on compressed spacetime latents: a video VAE shrinks time and space, and a diffusion transformer denoises spacetime patches.',
      'Tokens grow with length × resolution, and full attention grows with their square, which is why clip length and resolution are so expensive.',
      'Generating frames together keeps them consistent; independent frames flicker, and frame-by-frame extension drifts.',
      'Evaluate with distribution metrics (FVD), text alignment and human preference, and check physics and object permanence.',
      'Realistic synthetic video needs consent policies and provenance: content credentials and watermarks.'
    ],
    quiz: [
      {"type":"mcq","q":"Why do video diffusion models work in a compressed latent space rather than on pixels?","options":["Pixels cannot be denoised","Raw pixels would give far too many tokens for attention to handle","Latents look better than pixels","The text encoder only understands latents"],"answer":1,"explain":"A few seconds of video hold hundreds of millions of numbers. A video VAE shrinks them about 256-fold, and attention cost falls with the square of the token count."},
      {"type":"fill","q":"A 10-second clip at 24 frames per second is compressed 4× in time by the VAE. How many latent frames does it have?","answer":["60"],"explain":"240 frames / 4 = 60 latent frames."},
      {"type":"mcq","q":"What does factorized space-time attention save, and what does it cost?","options":["Nothing; it is only a different name for full attention","It compares far fewer token pairs, but information between distant frames travels only indirectly","It removes the need for a VAE","It makes clips longer automatically"],"answer":1,"explain":"Attending within frames and then across time costs T·S² + S·T² pairs instead of (T·S)², at the price of weaker direct links between distant frames."},
      {"type":"tf","q":"FVD compares the distribution of features of real and generated clips, and a lower FVD is better.","answer":true,"explain":"It is the video counterpart of FID: the Fréchet distance between Gaussian fits of the two sets of features from a pretrained video network."},
      {"type":"mcq","q":"How do video language models usually answer questions about a long video?","options":["They process every frame at full resolution","They sample a limited number of frames, encode each one and give the frame tokens to a language model","They read the subtitles only","They convert the video into audio"],"answer":1,"explain":"Sampling frames keeps the token count manageable, at the price of possibly missing fast events between the sampled frames."},
      {"type":"mcq","q":"Which measure helps people tell where a video came from?","options":["A higher resolution","C2PA content credentials and invisible watermarks","A longer prompt","Lower FVD"],"answer":1,"explain":"Signed provenance metadata and watermarks help identify generated media, although metadata can be stripped and watermarks weakened, so neither is absolute proof."}
    ],
    problems: [
      {
        id: 'video-token-count', title: 'Tokens in a Video Clip', difficulty: 'easy',
        prompt: `A video diffusion transformer does not see pixels. A **video VAE** first compresses the clip, by \`t_comp\` in time and \`s_comp\` in each spatial direction, and the latent is then cut into square patches of \`patch\` × \`patch\` latent positions, one token each. Implement \`video_tokens(frames, height, width, t_comp=4, s_comp=8, patch=2)\`:

- latent frames = frames / t_comp, **rounded up** (a partial group of frames still makes a latent frame);
- tokens per latent frame = (height // (s_comp · patch)) × (width // (s_comp · patch));
- return latent frames × tokens per latent frame.

\`\`\`text
video_tokens(120, 720, 1280)   ->  108000     5 s of 720p at 24 frames per second
video_tokens(1, 512, 512)      ->  1024       a single image
\`\`\``,
        starter: `
import math

def video_tokens(frames, height, width, t_comp=4, s_comp=8, patch=2):
    # TODO: latent frames (rounded up) times patches per latent frame
    pass
`,
        solution: `
import math

def video_tokens(frames, height, width, t_comp=4, s_comp=8, patch=2):
    latent_frames = math.ceil(frames / t_comp)
    per_frame = (height // (s_comp * patch)) * (width // (s_comp * patch))
    return latent_frames * per_frame
`,
        hints: ['math.ceil(frames / t_comp) rounds the number of latent frames up.', 'Each spatial direction shrinks by s_comp (the VAE) and then by patch (the patching).', '720 // 16 = 45 and 1280 // 16 = 80, so a 720p latent frame has 3,600 tokens.'],
        explain: `Doubling the length doubles the tokens, and doubling the resolution in both directions quadruples them. Full attention compares every token with every other, so its cost grows with the square: that is why long, high-resolution clips are so expensive to generate.`,
        tests: [
          { call: 'video_tokens(120, 720, 1280)', expect: '108000' },
          { name: 'a single image', call: 'video_tokens(1, 512, 512)', expect: '1024' },
          { name: 'a partial group of frames still counts', call: 'video_tokens(121, 720, 1280)', expect: '111600' },
          { name: '10 s at 1080p (1088 rows)', call: 'video_tokens(240, 1088, 1920)', hidden: true },
          { name: 'other compression settings', call: 'video_tokens(48, 256, 256, t_comp=2, s_comp=4, patch=4)', hidden: true }
        ]
      },
      {
        id: 'spacetime-attention-cost', title: 'Full versus Factorized Attention', difficulty: 'easy',
        prompt: `A clip has \`T\` latent frames with \`S\` tokens each, so N = T · S tokens. **Full spacetime attention** compares every token with every other: N × N pairs. **Factorized attention** first attends within each frame (T frames × S × S pairs) and then across time at each spatial position (S positions × T × T pairs).

Implement \`attention_pairs(T, S)\` returning the tuple \`(full, factorized)\`.

\`\`\`text
attention_pairs(2, 4)   ->  (64, 48)        full: 8 × 8;  factorized: 2·4·4 + 4·2·2
\`\`\``,
        starter: `
def attention_pairs(T, S):
    # TODO: return (full, factorized)
    pass
`,
        solution: `
def attention_pairs(T, S):
    n = T * S
    full = n * n
    factorized = T * S * S + S * T * T
    return full, factorized
`,
        hints: ['Full attention: (T · S) squared.', 'Factorized: spatial attention costs T · S², temporal attention costs S · T².'],
        explain: `For a 5-second 720p clip (T = 30, S = 3,600), full attention needs about 11.7 billion pairs per layer and head, factorized attention about 0.39 billion: 30 times fewer. The price is that information has to travel between frames in two hops, which makes long-range consistency harder; many recent models pay for full attention anyway.`,
        tests: [
          { call: 'attention_pairs(2, 4)', expect: '(64, 48)' },
          { name: 'a 5-second 720p clip', call: 'attention_pairs(30, 3600)', expect: '(11664000000, 392040000)' },
          { name: 'factorized is cheaper for long clips', call: 'attention_pairs(60, 3600)[0] > 20 * attention_pairs(60, 3600)[1]', expect: 'True', hidden: true },
          { name: 'another size', call: 'attention_pairs(16, 256)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ho et al., Video Diffusion Models (2022)', url: 'https://arxiv.org/abs/2204.03458' },
      { title: 'OpenAI, Video generation models as world simulators (Sora technical report, 2024)', url: 'https://openai.com/index/video-generation-models-as-world-simulators/' },
      { title: 'Unterthiner et al., Towards Accurate Generative Models of Video: FVD (2018)', url: 'https://arxiv.org/abs/1812.01717' },
      { title: 'C2PA, Content Credentials', url: 'https://c2pa.org/' }
    ]
  },

  // =====================================================================================
  {
    id: 'world-models', branch: 'spec', title: 'World Models: Learning to Predict, Imagine and Plan', short: 'World Models', level: 3, minutes: 50, prereqs: ['deep-rl', 'diffusion-models'],
    tags: ['world model', 'model-based rl', 'dynamics model', 'planning', 'model predictive control', 'mpc', 'random shooting', 'cem', 'dreamer', 'muzero', 'latent dynamics', 'imagination', 'genie', 'jepa', 'compounding error', 'ensemble'],
    summary: 'Models that learn how an environment responds to actions, and what they are used for: planning with model-predictive control, learning behaviour in imagination (Dreamer), planning with learned latent models (MuZero), interactive video world models, and predicting in representation space (JEPA); why errors compound over long rollouts, and how replanning and ensembles keep them in check.',
    why: `An agent that can predict what will happen if it acts can try actions in its head before trying them in the world. That makes learning far more data-efficient (a robot cannot crash a thousand times to learn), allows planning towards new goals without retraining, and resembles how people seem to think ahead. World models connect reinforcement learning, generative video and robotics, and they are one of the main bets on how AI systems will move beyond text.`,
    objectives: [
      'Define a world model, and contrast model-based with model-free reinforcement learning',
      'Fit a dynamics model from transitions and plan with it using model-predictive control',
      'Explain why prediction errors compound over long rollouts, and how replanning, short horizons and ensembles help',
      'Describe latent world models (Dreamer, MuZero), interactive video world models and representation-space prediction (JEPA)'
    ],
    sections: [
      { type: 'text', title: "Predicting the consequences of actions", md: `A **world model** predicts how an environment changes when an agent acts:

$$ ŝ_{t+1} = f_θ(s_t, a_t)          r̂_t = g_θ(s_t, a_t) $$
:::where
s_t — the state of the environment at time t (positions and velocities, or a learned summary of camera images)
a_t — the action taken at time t
f_θ — the learned dynamics model, with parameters θ; ŝ_{t+1} is its prediction of the next state
g_θ — an optional learned reward model; r̂_t is its prediction of the reward
:::

The reinforcement-learning methods of [Deep Reinforcement Learning](#/concept/deep-rl) are mostly **model-free**: they learn a policy or a value function directly from experience, and need a great deal of it. **Model-based** methods first learn f from experience and then use it:

- to **plan**: try candidate action sequences inside the model and pick the best one;
- to **imagine**: generate synthetic experience on which a policy is trained;
- to **pursue new goals** without new training, since the model describes the world, not one task.

The benefits are data efficiency and flexibility. The risk is that the model is wrong, and a planner that searches hard for good actions will find exactly the places where the model is wrong in its favour.` },
      { type: 'code', title: "Learn a model of a cart, then plan with it", md: `A cart on a track: its state is (position, velocity), friction slows it, and the action is a push between −1 and 1. The agent does not know the physics. It watches 200 random pushes, fits a linear world model by least squares, and then plans with it to stop the cart at position 1:`, code: `import numpy as np
rng = np.random.default_rng(0)

A_true = np.array([[1.0, 0.1], [0.0, 0.95]])        # state = (position, velocity): friction slows the cart
B_true = np.array([[0.0], [0.1]])                    # the action pushes it, with a force between -1 and 1
def step(s, a):                                      # the real world, with a little noise
    return A_true @ s + B_true @ np.atleast_1d(a) + rng.normal(0, 0.002, 2)

# 1. Collect experience with random actions
S, U, S2 = [], [], []
s = np.zeros(2)
for _ in range(200):
    a = rng.uniform(-1, 1)
    s2 = step(s, a)
    S.append(s); U.append([a]); S2.append(s2)
    s = s2
S, U, S2 = map(np.array, (S, U, S2))

# 2. Fit the world model s' = A s + B a by least squares
W, *_ = np.linalg.lstsq(np.hstack([S, U]), S2, rcond=None)
A_hat, B_hat = W[:2].T, W[2:].T
print("learned A:", (A_hat.round(3) + 0.0).tolist(), "  true A:", A_true.tolist())
print("learned B:", (B_hat.round(3) + 0.0).tolist(), "  true B:", B_true.tolist())

# 3. Plan with the model (random shooting), execute only the first action, then plan again (MPC)
goal = np.array([1.0, 0.0])
def plan(s, horizon=20, candidates=300):
    seqs = rng.uniform(-1, 1, (candidates, horizon))
    states = np.repeat(s[None, :], candidates, axis=0)
    cost = np.zeros(candidates)
    for t in range(horizon):
        states = states @ A_hat.T + seqs[:, t:t + 1] @ B_hat.T
        cost += ((states - goal) ** 2 * [1.0, 0.3]).sum(axis=1) + 0.01 * seqs[:, t] ** 2
    return seqs[cost.argmin(), 0]

s = np.zeros(2)
for t in range(100):
    s = step(s, plan(s))
print(f"after 100 steps: position {s[0]:.3f} (goal 1.0), velocity {s[1]:+.3f}")
print("the learned model gets the cart to the goal:", bool(abs(s[0] - 1) < 0.05 and abs(s[1]) < 0.05))`, after: `Two hundred random pushes are enough to recover the physics almost exactly, because the dynamics are simple. The planner never sees the true equations: at every step it tries 300 random 20-step plans inside the learned model, scores them by how close they bring the cart to the goal (and how gently), executes only the first push of the best plan, observes where the cart really is, and plans again. After 100 steps the cart sits at the goal and has almost stopped.`, predict: {"type":"mcq","q":"The agent fits its world model from only 200 random pushes and never sees the true physics. Will planning with the learned model bring the cart to the goal and stop it there?","options":["Yes: simple dynamics are learned almost exactly, and replanning at every step corrects small errors","No: 200 transitions are far too few for any model","Only if the planner is given the true equations","It reaches the goal but cannot stop"],"answer":0,"explain":"The learned matrices match the true ones to about three decimals, and model-predictive control replans from the real state at every step, so small model errors never accumulate.","expect":["the learned model gets the cart to the goal: True"]} },
      { type: 'text', title: "Planning with a learned model: MPC, random shooting and CEM", md: `The planner above is **model-predictive control (MPC)**: optimize a short sequence of actions inside the model, execute only the first, observe the real outcome, and optimize again from there. Replanning at every step is what makes it robust: the plan only has to be good for a few steps, and surprises are corrected as soon as they are seen.

How the action sequence is optimized:

- **Random shooting**: sample many random sequences and keep the best, as above. Simple, parallel and surprisingly effective for short horizons and few action dimensions.
- **The cross-entropy method (CEM)**: sample sequences from a Gaussian, keep the best 10%, refit the Gaussian to them, and repeat a few times. It concentrates the search where good plans are, and is the standard choice in model-based RL (for example in PETS, which pairs it with an ensemble of models).
- **Gradient-based planning**: when the model is differentiable, follow the gradient of the predicted cost with respect to the actions, as in trajectory optimization for robots.
- **Tree search**: for discrete actions, search a tree of possible futures, as MuZero does (below).` },
      { type: 'check', question: {"type":"mcq","q":"Why does model-predictive control execute only the first action of each plan and then plan again?","options":["To save computation","So that model errors and unexpected events are corrected at every step from the real state","Because the model can only predict one step","Because the later actions are always random"],"answer":1,"explain":"Every replan starts from the observed state, so the plan never has to be accurate far into the future, and deviations are corrected immediately."} },
      { type: 'code', title: "Errors compound over long rollouts", md: `How far ahead can a learned model be trusted? Fit two models from noisy random transitions, one from 20 and one from 200, and compare their open-loop predictions (the model predicts many steps ahead without seeing the real state again) with the truth:`, code: `import numpy as np
rng = np.random.default_rng(1)
A_true = np.array([[1.0, 0.1], [0.0, 0.95]])
B_true = np.array([[0.0], [0.1]])

def fit(n, noise=0.01):                               # fit a model from n noisy random transitions
    S = rng.uniform(-1, 1, (n, 2)); U = rng.uniform(-1, 1, (n, 1))
    S2 = S @ A_true.T + U @ B_true.T + rng.normal(0, noise, (n, 2))
    W, *_ = np.linalg.lstsq(np.hstack([S, U]), S2, rcond=None)
    return W[:2].T, W[2:].T

def rollout_error(A, B, horizon, trials=200):
    errs = np.zeros(horizon)
    for _ in range(trials):
        s_true = s_model = rng.uniform(-1, 1, 2)
        for t in range(horizon):
            a = rng.uniform(-1, 1, 1)
            s_true = A_true @ s_true + B_true @ a
            s_model = A @ s_model + B @ a
            errs[t] += np.abs(s_model - s_true)[0] / trials     # position error
    return errs

for n in (20, 200):
    A, B = fit(n)
    e = rollout_error(A, B, 50)
    print(f"model from {n:3d} transitions: position error after 1 step {e[0]:.4f}, 10 steps {e[9]:.3f}, 50 steps {e[49]:.3f}")
A, B = fit(20)
e = rollout_error(A, B, 50)
print(f"the 50-step error is {e[49] / e[0]:.0f} times the one-step error")
print("errors compound over long rollouts:", bool(e[49] > 10 * e[0]))`, after: `A one-step error of about a thousandth grows a hundredfold over 50 steps: each prediction starts from the previous, slightly wrong one, and a small error in the friction or the push strength is applied again at every step. More data shrinks the error but does not change its growth. This is why practical systems plan over short horizons and replan often, and why long imagined rollouts need care.`, predict: {"type":"mcq","q":"A learned model's one-step prediction error is about 0.001. Roughly how large is its error after 50 steps predicted without looking at the real state?","options":["About the same, 0.001","About 0.05, fifty times larger","Around a hundred times larger, or more","Smaller, because the errors cancel"],"answer":2,"explain":"Each step builds on the previous prediction, so errors in the dynamics are applied over and over: here the 50-step error is more than a hundred times the one-step error.","expect":["errors compound over long rollouts: True"]} },
      { type: 'chart', title: "Open-loop prediction error against the horizon", md: `The two models from the code above, step by step:`, after: `On a log scale the error rises steeply over the first steps and keeps growing, by two orders of magnitude over 50 steps for both models. Ten times more data lowers the whole curve but hardly changes its slope: better models delay the problem, and replanning is what solves it.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],"yLog":true,"points":false,"series":[{"name":"model fitted from 20 transitions","y":[0.001231,0.002417,0.003628,0.005077,0.006807,0.008826,0.01075,0.01288,0.01541,0.01805,0.02098,0.02412,0.02762,0.0313,0.03512,0.03935,0.04362,0.04822,0.05295,0.058,0.06327,0.06892,0.07471,0.08081,0.08723,0.09384,0.1006,0.1078,0.1151,0.1228,0.1307,0.1387,0.147,0.1555,0.1642,0.173,0.182,0.1914,0.2011,0.211,0.2213,0.232,0.2428,0.2539,0.2653,0.277,0.2892,0.3012,0.3136,0.3261]},{"name":"model fitted from 200 transitions","y":[0.0007749,0.001638,0.002549,0.003549,0.004572,0.005724,0.006843,0.007981,0.009301,0.01061,0.01192,0.01338,0.01484,0.01633,0.01791,0.01944,0.02096,0.02254,0.02429,0.02597,0.02772,0.02947,0.03135,0.03325,0.03518,0.03717,0.03918,0.04136,0.04348,0.04568,0.04796,0.05027,0.05266,0.05513,0.05762,0.06019,0.06289,0.0657,0.06861,0.07155,0.07459,0.07765,0.08084,0.08406,0.08731,0.09062,0.09401,0.09751,0.1011,0.1047]}],"xLabel":"steps predicted ahead without seeing the real state","yLabel":"average position error (log scale)"} },
      { type: 'text', title: "Living with model errors", md: `- **Short horizons and replanning** (MPC) keep errors from compounding far.
- **Ensembles**: train several models on resampled data. Where they disagree, the model is uncertain, so the planner can penalize those actions or avoid them; PETS and MBPO are well-known methods built on this idea.
- **Mixing real and imagined experience**: Dyna-style methods train a model-free learner on both real transitions and short rollouts from the model.
- **Learning where it matters**: collect new real data around the states the planner actually visits, rather than only random exploration.
- **Pessimism in offline settings**: when no new data can be collected, prefer actions the data supports, because the planner will otherwise exploit the model's errors.` },
      { type: 'text', title: "Learning in imagination: Dreamer", md: `Pixels are a poor state: they are huge, and most of their detail does not matter for control. **Dreamer** (Hafner and colleagues, from 2019; DreamerV3 in 2023) learns a compact **latent** world model from images, a recurrent state-space model that encodes each observation into a latent state and predicts the next latent state, the reward and whether the episode ends. Its policy and value function are trained entirely on **imagined** trajectories rolled out in the latent space, thousands at a time; only the world model is trained on real experience.

DreamerV3 uses one set of hyperparameters across very different domains, from Atari games to continuous control, and was the first algorithm to collect diamonds in Minecraft from scratch, without human demonstrations, a task that needs long sequences of correct decisions.` },
      { type: 'text', title: "Planning without predicting pixels: MuZero", md: `**MuZero** (DeepMind, 2019) goes further: its model never predicts observations at all. From a latent state and an action it predicts only what planning needs: the **reward**, the **value** of the position and a **policy** over the next actions. It then plans with Monte Carlo tree search, like AlphaZero, but without being told the rules of the game. MuZero matched AlphaZero at Go, chess and shogi and set new records on Atari games. The lesson is that a world model only has to be accurate about what matters for the decision, not about every detail of the world.` },
      { type: 'text', title: "Video world models and simulators", md: `A video generator that is conditioned on actions is a world model: given the frames so far and an action (a key press, a camera move, a steering angle), it predicts the next frames (see [Video Models](#/concept/video-generation)).

- **GameNGen** (Google, 2024) ran the game DOOM inside a diffusion model at about 20 frames per second: the model *was* the game engine, predicting each frame from the previous frames and the player's input.
- **Genie** (Google DeepMind, 2024 onwards) generates playable environments from an image or a text prompt, and learned what the "actions" are from unlabeled videos; Genie 3 (2025) was reported to generate interactive worlds in real time for several minutes.
- **Cosmos** (NVIDIA, 2025) is a family of world foundation models for "physical AI": generating realistic driving and robot scenarios to train and test policies.

Their limits are the ones of video generation, made worse by interaction: consistency over minutes, physics that holds when an agent probes it, precise control, and cost.` },
      { type: 'text', title: "Predicting representations: JEPA, and what language models know", md: `**Predict embeddings, not pixels.** Yann LeCun has argued that predicting every pixel wastes capacity on details that cannot be predicted (leaves in the wind, noise). Joint-embedding predictive architectures (**JEPA**) instead predict the *representation* of a missing or future part of the input from the representation of the visible context, much like the self-supervised methods in [Self-Supervised Learning](#/concept/self-supervised). V-JEPA (Meta, 2024) learns from video this way, and V-JEPA 2 (2025) was used to plan robot actions by comparing predicted and goal representations.

**Do language models have world models?** In a well-known experiment (Othello-GPT, 2023), a transformer trained only to predict legal moves of the board game Othello developed an internal representation of the board: simple probes could read the board state from its activations, and editing that representation changed its predictions. How far such internal models extend to the real world, and how reliable they are, is an open research question: large language models describe physics fluently but often fail when asked to predict the outcome of unusual physical situations.` },
      { type: 'check', question: {"type":"tf","q":"A world model whose one-step predictions are very accurate will also make accurate 50-step predictions.","answer":false,"explain":"Errors compound: each prediction starts from the previous one, so a small one-step error can grow a hundredfold over 50 steps. That is why planners replan often."} }
    ],
    when: `- Interaction with the real world is slow, costly or risky (robots, industrial control, healthcare), so every sample counts.
- Goals change often: a model can plan for a new goal without retraining a policy.
- You need simulated environments or training data for other agents (driving, robotics, games).
- Prefer model-free methods when interaction is cheap and the dynamics are too complex to model accurately.`,
    pitfalls: `- Trusting long open-loop rollouts: small one-step errors compound.
- Letting the planner exploit the model: it finds actions where the model is wrong and optimistic. Penalize uncertainty, keep horizons short, and replan.
- Modelling every pixel when only a few quantities matter for the decision.
- Evaluating a world model only by its prediction loss, rather than by how well the plans it produces work.`,
    takeaways: [
      'A world model predicts the next state (and reward) from the current state and action; planning searches over actions inside the model.',
      'Model-predictive control plans a short sequence, executes only the first action, observes and plans again, which corrects model errors.',
      'Errors compound over long rollouts: a tiny one-step error can grow a hundredfold over 50 steps.',
      'Dreamer learns behaviour inside a latent model, MuZero plans with a model of rewards and values only, and video world models make environments that can be played.',
      'JEPA-style models predict representations rather than pixels, ignoring details that cannot be predicted.'
    ],
    quiz: [
      {"type":"mcq","q":"What is the main advantage of model-based over model-free reinforcement learning?","options":["It never needs any real data","It usually needs far less real experience, and the model can be reused for new goals","It always finds better policies","It does not need a reward"],"answer":1,"explain":"Planning or imagining inside a learned model makes each real interaction go further, which matters when interaction is slow, costly or risky."},
      {"type":"mcq","q":"What does MuZero's learned model predict?","options":["The next image","Rewards, values and a policy, from latent states, but not observations","The rules of the game, written as code","Only whether the game is over"],"answer":1,"explain":"MuZero's model is trained to predict only the quantities needed for tree search, so it never has to reconstruct observations."},
      {"type":"tf","q":"Dreamer trains its policy on trajectories imagined by its learned latent world model.","answer":true,"explain":"Only the world model is trained on real experience; the actor and critic learn from imagined latent rollouts."},
      {"type":"mcq","q":"A planner keeps choosing actions that look excellent in the model but fail in reality. What helps most?","options":["Planning over longer horizons","An ensemble of models, penalizing actions where they disagree, plus short horizons and replanning","Removing the reward model","Training the model for fewer steps"],"answer":1,"explain":"The planner is exploiting model errors. Disagreement between ensemble members flags where the model is unreliable, and short horizons limit how far the errors can grow."},
      {"type":"fill","q":"Random shooting evaluates 100 candidate action sequences, each 20 steps long. How many model predictions does one round of planning make?","answer":["2000","2,000"],"explain":"100 sequences × 20 steps = 2,000 one-step predictions, which is why planners batch them on a GPU."},
      {"type":"mcq","q":"What does a JEPA-style model predict?","options":["Every pixel of the next frame","The representation (embedding) of a missing or future part of the input","A text caption","The reward only"],"answer":1,"explain":"Predicting in representation space lets the model ignore unpredictable details and focus on what can be predicted."}
    ],
    problems: [
      {
        id: 'fit-linear-dynamics', title: 'Fit a Linear World Model', difficulty: 'medium',
        prompt: `A **world model** for a simple system predicts the next state from the current state and action: s' = A s + B a. Given transitions as NumPy arrays \`states\` (n × d), \`actions\` (n × m) and \`next_states\` (n × d), implement \`fit_dynamics(states, actions, next_states)\` that returns the least-squares estimates \`(A, B)\` with shapes (d × d) and (d × m).

Stack the inputs side by side, X = [states, actions], solve X W ≈ next_states with \`np.linalg.lstsq\`, and split W: its first d rows are Aᵀ and its remaining rows are Bᵀ.`,
        starter: `
import numpy as np

def fit_dynamics(states, actions, next_states):
    # TODO: least squares on [states, actions] -> next_states, then split into A and B
    pass
`,
        solution: `
import numpy as np

def fit_dynamics(states, actions, next_states):
    d = states.shape[1]
    X = np.hstack([states, actions])
    W, *_ = np.linalg.lstsq(X, next_states, rcond=None)
    return W[:d].T, W[d:].T
`,
        hints: ['X = np.hstack([states, actions]) has d + m columns.', 'np.linalg.lstsq(X, next_states, rcond=None)[0] is W, with shape (d + m) × d.', 'A = W[:d].T and B = W[d:].T.'],
        explain: `With exact (noise-free) data and enough varied transitions, least squares recovers the true dynamics. With noise it gives the best linear fit, and its errors are what compound when the model is rolled out many steps ahead. Real world models are neural networks, but they are fitted the same way: by regressing the next state on the current state and action.`,
        tests: [
          { name: 'recovers A', setup: 'import numpy as np\nrng = np.random.default_rng(0)\nA0 = np.array([[1.0, 0.1], [0.0, 0.95]]); B0 = np.array([[0.0], [0.1]])\nS = rng.uniform(-1, 1, (50, 2)); U = rng.uniform(-1, 1, (50, 1)); S2 = S @ A0.T + U @ B0.T', call: 'np.round(fit_dynamics(S, U, S2)[0], 6).tolist()', expect: '[[1.0, 0.1], [0.0, 0.95]]' },
          { name: 'recovers B', setup: 'import numpy as np\nrng = np.random.default_rng(0)\nA0 = np.array([[1.0, 0.1], [0.0, 0.95]]); B0 = np.array([[0.0], [0.1]])\nS = rng.uniform(-1, 1, (50, 2)); U = rng.uniform(-1, 1, (50, 1)); S2 = S @ A0.T + U @ B0.T', call: 'np.round(fit_dynamics(S, U, S2)[1], 6).tolist()', expect: '[[0.0], [0.1]]' },
          { name: 'shapes with two actions', setup: 'import numpy as np\nrng = np.random.default_rng(1)\nS = rng.normal(size=(40, 3)); U = rng.normal(size=(40, 2)); S2 = rng.normal(size=(40, 3))', call: '[m.shape for m in fit_dynamics(S, U, S2)]', expect: '[(3, 3), (3, 2)]' },
          { name: 'noisy data vs reference', setup: 'import numpy as np\nrng = np.random.default_rng(2)\nS = rng.normal(size=(80, 2)); U = rng.normal(size=(80, 1)); S2 = S @ np.array([[0.9, 0.2], [-0.1, 0.8]]).T + U @ np.array([[0.5], [0.3]]).T + rng.normal(0, 0.05, (80, 2))', call: 'np.round(np.hstack(fit_dynamics(S, U, S2)), 4).tolist()', hidden: true }
        ]
      },
      {
        id: 'best-plan', title: 'Pick the Best Plan', difficulty: 'easy',
        prompt: `**Random shooting** plans by rolling candidate action sequences through the world model and keeping the best one. Implement \`best_plan(A, B, s0, goal, plans)\`:

- \`A\` is a d × d list of lists, \`B\` a list of d numbers (one action dimension), \`s0\` and \`goal\` lists of d numbers;
- \`plans\` is a list of action sequences (lists of numbers);
- roll each plan out with s ← A s + B · a, one action at a time, starting from s0;
- return the **index** of the plan whose final state is closest to \`goal\` (Euclidean distance); on a tie, the lowest index.

\`\`\`text
A = [[1, 0], [0, 1]], B = [1, 0], s0 = [0, 0], goal = [3, 0]
plans = [[1, 1, 1], [2, 2, 2], [0, 0, 0]]   ->  0      (final states [3, 0], [6, 0], [0, 0])
\`\`\``,
        starter: `
import numpy as np

def best_plan(A, B, s0, goal, plans):
    # TODO: roll out every plan with the model and return the index of the best one
    pass
`,
        solution: `
import numpy as np

def best_plan(A, B, s0, goal, plans):
    A, B, goal = np.array(A, dtype=float), np.array(B, dtype=float), np.array(goal, dtype=float)
    best, best_dist = 0, float('inf')
    for i, plan in enumerate(plans):
        s = np.array(s0, dtype=float)
        for a in plan:
            s = A @ s + B * a
        dist = float(np.linalg.norm(s - goal))
        if dist < best_dist:
            best, best_dist = i, dist
    return best
`,
        hints: ['Convert A, B, s0 and goal to NumPy arrays first.', 'For each action: s = A @ s + B * a.', 'Keep the best distance seen so far and update only when a plan is strictly better, so ties keep the lowest index.'],
        explain: `This is the heart of random shooting: sample many action sequences, evaluate them inside the model, keep the best. In model-predictive control you then execute only its first action, observe the real state and plan again, which keeps model errors from piling up.`,
        tests: [
          { call: 'best_plan([[1, 0], [0, 1]], [1, 0], [0, 0], [3, 0], [[1, 1, 1], [2, 2, 2], [0, 0, 0]])', expect: '0' },
          { name: 'another goal', call: 'best_plan([[1, 0], [0, 1]], [1, 0], [0, 0], [5, 0], [[1, 1, 1], [2, 2, 2], [0, 0, 0]])', expect: '1' },
          { name: 'ties keep the lowest index', call: 'best_plan([[1, 0], [0, 1]], [1, 0], [0, 0], [3, 0], [[2, 2], [1, 1], [3, 3]])', expect: '0' },
          { name: 'dynamics with velocity', call: 'best_plan([[1, 0.1], [0, 0.95]], [0, 0.1], [0, 0], [0.05, 0], [[1] * 10, [0.5] * 10, [0] * 10, [-1] * 10])', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ha & Schmidhuber, World Models (2018)', url: 'https://worldmodels.github.io/' },
      { title: 'Hafner et al., Mastering Diverse Domains through World Models (DreamerV3, 2023)', url: 'https://arxiv.org/abs/2301.04104' },
      { title: 'Schrittwieser et al., Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero, 2019)', url: 'https://arxiv.org/abs/1911.08265' }
    ]
  },

  // =====================================================================================
  {
    id: 'robotics-vla', branch: 'spec', title: 'Robot Learning & Vision-Language-Action Models', short: 'Robotics & VLAs', level: 3, minutes: 50, prereqs: ['multimodal-models', 'deep-rl'],
    tags: ['robotics', 'robot learning', 'imitation learning', 'behaviour cloning', 'behavior cloning', 'dagger', 'covariate shift', 'teleoperation', 'vla', 'vision-language-action', 'rt-2', 'openvla', 'action chunking', 'diffusion policy', 'sim-to-real', 'domain randomization', 'open x-embodiment', 'embodiment'],
    summary: 'How machine learning controls robots: the classical sense–plan–act pipeline, imitation learning and why behaviour cloning drifts (and how DAgger and action chunking help), why averaging demonstrations fails, reinforcement learning in simulation and the sim-to-real gap, and vision-language-action models that turn a pretrained vision-language model into a robot policy; data, evaluation and safety.',
    why: `Language models learned from the whole internet; robots have no such dataset, must act in real time, and their mistakes have physical consequences. Robot learning is where data efficiency, generalization and safety meet hardest. Vision-language-action models are the newest attempt to borrow the broad knowledge of web-scale models for physical tasks, and knowing their strengths and limits is essential for following "physical AI".`,
    objectives: [
      'Describe the classical robotics pipeline and where learning replaces or complements it',
      'Explain behaviour cloning, why its errors compound (covariate shift), and how DAgger and action chunking reduce them',
      'Explain how vision-language-action models turn a pretrained vision-language model into a policy, including action tokens and action experts',
      'Reason about robot data, the sim-to-real gap, evaluation with confidence intervals, and safety'
    ],
    sections: [
      { type: 'text', title: "Why robots are hard for machine learning", md: `Language models learned from trillions of words that already existed. Robots have no such resource:

- **Data is scarce and expensive.** There is no internet of robot actions. Every demonstration has to be performed, usually by a person teleoperating the robot, at the speed of the real world.
- **Bodies differ.** Each robot has its own arms, grippers, cameras and controllers, so data from one robot does not transfer directly to another.
- **Time is real.** Control loops run 10 to 1,000 times per second, and a late action is a wrong action.
- **The world is partially observed and endlessly varied:** occlusions, reflections, deformable objects, new kitchens and new lighting.
- **Mistakes are physical.** A wrong action can break things or hurt people, which limits trial and error.

*Moravec's paradox* summarizes the difficulty: what is easy for people (picking up a cup, walking on uneven ground) has proved much harder for machines than what is hard for people (chess, calculus).` },
      { type: 'text', title: "Sense, plan, act, and where learning fits", md: `The classical robotics pipeline splits the problem into modules:

1. **Perception**: cameras, depth sensors and force sensors, turned into object positions and a map.
2. **State estimation**: combining noisy measurements over time into the robot's best guess of where everything is.
3. **Planning**: finding a collision-free path or grasp.
4. **Control**: following the plan with feedback controllers, such as PID controllers or model-predictive control (see [World Models](#/concept/world-models)).

Learning entered module by module: learned perception (detection and segmentation, see [Object Detection & Segmentation](#/concept/detection-segmentation)), learned grasp predictors, learned dynamics models. The current trend is **end-to-end visuomotor policies** that map camera images (and an instruction) directly to motor commands. Policies are trained in three main ways:

- **Imitation learning** from demonstrations, the dominant approach for manipulation;
- **Reinforcement learning**, mostly in simulation, where trial and error is free (locomotion, in-hand manipulation);
- **Pretrained foundation models** fine-tuned for control: vision-language-action models.` },
      { type: 'text', title: "Imitation learning and covariate shift", md: `**Behaviour cloning** treats imitation as supervised learning: collect (observation, action) pairs from an expert and train a policy to predict the expert's action. It is simple and often works, but it has a structural flaw. The expert's demonstrations only cover the states the expert visits. The learned policy makes small mistakes, which take it to states the expert never visited, where its predictions are worse, which takes it further away. This is **covariate shift**: the training and test distributions of states differ because the policy itself changes which states it sees.

Ross and Bagnell showed that the errors compound with the length of the task:

$$ cost of behaviour cloning ≲ T² · ε          cost with DAgger ≲ T · ε (roughly) $$
:::where
T — the number of steps in the task
ε — the policy's error rate on the states it was trained on
T² · ε — how the expected cost of behaviour cloning can grow: an early mistake affects every later step
T · ε — the roughly linear growth when the training data covers the states the learner itself visits
:::

**DAgger** (dataset aggregation) closes the gap: run the learned policy, ask the expert which action it *would* have taken in each state the learner visited, add those labels to the dataset, retrain, and repeat. The data then covers the learner's own mistakes and how to recover from them. In practice, people get a similar effect by deliberately demonstrating recoveries, injecting noise during demonstrations, or letting an operator take over and correct the robot.` },
      { type: 'code', title: "Behaviour cloning drifts, DAgger recovers", md: `A robot must keep its sideways offset from a line near zero. The system is unstable, so it needs constant small corrections. The expert's demonstrations are calm and stay very close to the line, but the real world is bumpier. The policy simply copies the expert's action from the most similar state it has seen:`, code: `import numpy as np
rng = np.random.default_rng(0)

# A robot keeps its sideways offset x from a line at 0. Left alone the offset grows (the system is unstable),
# so the policy must keep correcting it. The expert's correction is a = -3x.
def step(x, a, noise):
    return x + 0.1 * (x + a) + rng.normal(0, noise)
expert = lambda x: -3.0 * x

def fit_policy(X, Y):                               # copy the action of the most similar state in the data
    order = np.argsort(X)
    Xs, Ys = np.asarray(X)[order], np.asarray(Y)[order]
    def policy(x):
        i = int(np.clip(np.searchsorted(Xs, x), 1, len(Xs) - 1))
        return Ys[i] if abs(Xs[i] - x) < abs(Xs[i - 1] - x) else Ys[i - 1]
    return policy

def run(policy, steps=100, noise=0.03):             # the real world is bumpier than the demonstrations were
    x, xs = 0.0, []
    for _ in range(steps):
        x = step(x, policy(x), noise)
        xs.append(x)
        if abs(x) > 1:                              # left the safe zone: the episode fails
            break
    return xs

# Behaviour cloning: 20 calm expert demonstrations, so the data only covers small offsets
X, Y = [], []
for _ in range(20):
    x = 0.0
    for _ in range(100):
        X.append(x); Y.append(expert(x))
        x = step(x, expert(x), 0.005)
bc = fit_policy(X, Y)

# DAgger: run the current policy, ask the expert what it would have done in the states the LEARNER reached, retrain
Xd, Yd, policy = list(X), list(Y), bc
for _ in range(4):
    for _ in range(10):
        for x in run(policy):
            Xd.append(x); Yd.append(expert(x))
    policy = fit_policy(Xd, Yd)
dagger = policy

print(f"demonstrations cover offsets from {min(X):+.3f} to {max(X):+.3f}")
rate = {}
for name, pol in (("expert", expert), ("behaviour cloning", bc), ("DAgger", dagger)):
    runs = [run(pol) for _ in range(200)]
    rate[name] = np.mean([len(r) == 100 for r in runs])
    print(f"{name:18s} stays in the safe zone for 100 steps in {rate[name]:4.0%} of 200 runs")
print("DAgger succeeds far more often than behaviour cloning:", bool(rate["DAgger"] > rate["behaviour cloning"] + 0.3))`, after: `The cloned policy is perfect on the states it has seen, but the demonstrations only cover offsets up to about ±0.03. The first bump that pushes the robot further leaves it copying actions meant for smaller offsets, which are too weak to bring it back, and it drifts out of the safe zone in most runs. DAgger collects expert labels exactly where the learner goes wrong, and after four rounds the policy is as reliable as the expert. The expert never needed to demonstrate more often, only in the right places.`, predict: {"type":"mcq","q":"The cloned policy copies the expert perfectly on every state in its 20 calm demonstrations. In the bumpier real world, how often will it keep the robot in the safe zone for 100 steps?","options":["Always, like the expert","In most runs, but not all","Only in a minority of runs","Never, not even for one step"],"answer":2,"explain":"About 18% of runs succeed. Bumps push the robot to offsets the demonstrations never covered, where the copied corrections are too weak, so it drifts away. DAgger, trained on the states the learner actually reaches, succeeds every time.","expect":["DAgger succeeds far more often than behaviour cloning: True"]} },
      { type: 'chart', title: "Share of runs still in the safe zone", md: `The policies from the code above, each run 400 more times:`, after: `Behaviour cloning loses runs steadily from the first few dozen steps on, each time a bump pushes the robot beyond the states its demonstrations covered, and only about 16% last all 100 steps. The DAgger policy stays with the expert throughout. A longer task would lose even more of the cloned runs, which is the compounding of errors in practice.`, chart: {"kind":"line","x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100],"points":false,"series":[{"name":"expert","y":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],"dash":true},{"name":"behaviour cloning (20 calm demonstrations)","y":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,99.8,99.8,99.8,99.8,99.8,99.5,98.5,98.2,97,95.5,94.5,93.2,91.5,89.5,87.8,87,85.8,84.2,82.8,81.2,79.5,77.8,75.8,73.8,71,70.5,68.8,66.8,65.2,64.2,63,60.5,58.5,56,55.2,53.8,53,50.7,49.8,48.2,47.5,46.2,44.5,42.5,40.8,40.2,39.2,38.8,38.2,37.8,36.5,34.8,34,32.2,31.2,30.2,29.5,28.2,27.8,27.3,26.5,26,25.2,25.2,24.2,24,23.5,23,21.8,21.8,21.2,20.2,20,18.8,17.2,17.2,16.2,16.2,16,15.5]},{"name":"DAgger (after 4 rounds)","y":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]}],"xLabel":"step","yLabel":"runs still in the safe zone (%)","yMin":0,"yMax":100} },
      { type: 'code', title: "Averaging two good demonstrations gives a bad one", md: `A second, separate problem: demonstrations are often **multimodal**. Faced with an obstacle straight ahead, half the demonstrators steer left and half steer right. Both are fine. What does a policy trained to minimize squared error learn?`, code: `import numpy as np
rng = np.random.default_rng(0)
# An obstacle sits straight ahead. Half the demonstrations steer left (-1), half steer right (+1).
demos = np.where(rng.random(200) < 0.5, -1.0, 1.0) + rng.normal(0, 0.05, 200)

mse_policy = demos.mean()                            # what a regression trained with squared error predicts
samples = rng.choice(demos, 1000)                    # a generative policy (like a diffusion policy) samples a demo-like action
hits = lambda a: np.abs(a) < 0.3                     # steering less than 0.3 either way hits the obstacle
print(f"squared-error policy steers {mse_policy:+.2f}: hits the obstacle = {bool(hits(mse_policy))}")
print(f"sampling policy: hits the obstacle in {hits(samples).mean():.0%} of 1000 tries")
print("averaging two good demonstrations gives a bad action:", bool(hits(mse_policy)))`, after: `Squared error is minimized by the **average** of the demonstrations, so the regression policy steers almost straight ahead, into the obstacle, an action no demonstrator ever took. A policy that *samples* an action from the distribution of demonstrations always picks one of the two valid manoeuvres.`, predict: {"type":"mcq","q":"Half the demonstrations steer left around an obstacle and half steer right. What does a policy trained with squared error do?","options":["Steers left, the first option it saw","Steers right, the more common option","Steers roughly straight ahead, into the obstacle","Refuses to move"],"answer":2,"explain":"Squared error is minimized by the average action, and the average of −1 and +1 is about 0: straight ahead.","expect":["averaging two good demonstrations gives a bad action: True"]} },
      { type: 'text', title: "Action chunking and diffusion policies", md: `Two ideas from 2023 made imitation learning far more reliable on real robots:

- **Action chunking** (ACT, from the ALOHA bimanual robot project): predict a *chunk* of the next k actions, for example 50 to 100 steps, instead of one action at a time, and execute them smoothly (often averaging overlapping chunks). Fewer decisions means fewer chances for errors to compound, and a chunk can represent pauses and multi-step motions that a single-step policy finds hard.
- **Diffusion policies**: generate the action chunk with a diffusion model (see [Diffusion Models & Flow Matching](#/concept/diffusion-models)) conditioned on the observation. A generative model represents multimodal behaviour, left *or* right, instead of averaging it, and handles high-dimensional actions well.

Both have become standard ingredients of modern robot policies, including the action experts of vision-language-action models.` },
      { type: 'text', title: "Vision-language-action models", md: `A **vision-language-action model** (VLA) starts from a vision-language model pretrained on web images and text (see [Multimodal Models](#/concept/multimodal-models)) and is fine-tuned on robot trajectories, so that given camera images and an instruction ("put the apple in the bowl") it outputs actions. The hope is to inherit the broad knowledge of the web: what an apple is, which object is "the extinct animal", what "tidy up" means.

Two main designs:

- **Actions as tokens.** RT-2 (Google DeepMind, 2023) discretized each action dimension into 256 bins and wrote actions as text tokens, so the language model could predict them like words. Trained on robot data together with web data, it carried out instructions involving objects and concepts never seen in its robot data. **OpenVLA** (2024) is an open 7-billion-parameter model in the same style, trained on about 970,000 robot episodes.
- **An action expert.** π0 (Physical Intelligence, 2024) attaches a smaller "action expert" to the vision-language backbone. It generates chunks of continuous actions with flow matching, at up to 50 actions per second, for dexterous tasks such as folding laundry. Later models from several companies, including π0.5, Google DeepMind's Gemini Robotics and NVIDIA's GR00T models for humanoids, follow similar ideas.

Turning a continuous action into a token, and back:

$$ k = round( (a − a_min) / (a_max − a_min) · (K − 1) )          error ≤ (a_max − a_min) / (2 (K − 1)) $$
:::where
a — one continuous action value, such as a joint velocity or the gripper opening
a_min, a_max — the range of that action dimension
K — the number of bins (tokens) per dimension, 256 in RT-2
k — the token index, from 0 to K − 1
error — the largest difference between an action and the value its token is decoded back to: half a bin
:::` },
      { type: 'code', title: "Turning actions into tokens", code: `import numpy as np
rng = np.random.default_rng(0)
low, high = -1.0, 1.0
def to_tokens(a, bins=256):                          # RT-2 style: each action dimension becomes one of \`bins\` tokens
    return np.round((np.clip(a, low, high) - low) / (high - low) * (bins - 1)).astype(int)
def from_tokens(tok, bins=256):
    return low + tok / (bins - 1) * (high - low)

actions = rng.uniform(low, high, (10_000, 7))        # 7 numbers per step: x, y, z, roll, pitch, yaw, gripper
print("bins   largest error   typical (RMS) error")
for bins in (8, 32, 256, 1024):
    err = np.abs(from_tokens(to_tokens(actions, bins), bins) - actions)
    print(f"{bins:5d} {err.max():14.4f} {np.sqrt((err ** 2).mean()):16.4f}")
err256 = np.abs(from_tokens(to_tokens(actions)) - actions).max()
print("with 256 bins the error is at most half a bin:", bool(err256 <= (high - low) / 255 / 2 + 1e-12))
print("tokens per second for a 7-number action at 5 Hz:", 7 * 5, "  at 50 Hz:", 7 * 50)`, after: `With 256 bins the largest rounding error is about 0.004 on a range of 2, precise enough for most arm motions, and quadrupling the bins quarters the error. The real cost is speed: one token per action dimension means 35 tokens per second at 5 Hz but 350 at 50 Hz, each produced by a multi-billion-parameter model. That is why fast, dexterous control moved to action experts that output whole chunks of continuous actions at once.`, predict: {"type":"mcq","q":"Actions in the range [−1, 1] are written as one of 256 tokens. What is the largest error introduced by rounding to a token?","options":["About 0.1","About 0.01","About 0.004","Zero"],"answer":2,"explain":"Neighbouring tokens are 2 / 255 ≈ 0.0078 apart, and rounding moves a value by at most half of that: about 0.0039.","expect":["with 256 bins the error is at most half a bin: True"]} },
      { type: 'chart', title: "Rounding error against the number of action bins", md: `Exact values for actions spread evenly over [−1, 1], from the formula above:`, after: `Both errors halve every time the number of bins doubles: a straight line on log-log axes. At 256 bins the largest error is about 0.004, small next to the noise of most robot motions, so the real costs of action tokens are speed and vocabulary size, not precision.`, chart: {"kind":"line","x":[4,8,16,32,64,128,256,512,1024],"xLog":true,"yLog":true,"series":[{"name":"largest error: half a bin","y":[0.3333,0.1429,0.06667,0.03226,0.01587,0.007874,0.003922,0.001957,0.0009775]},{"name":"typical (RMS) error for uniform actions","y":[0.1925,0.08248,0.03849,0.01862,0.009164,0.004546,0.002264,0.00113,0.0005644]}],"xLabel":"bins per action dimension (log scale)","yLabel":"error on an action range of 2 (log scale)","vlines":[{"x":256,"label":"RT-2: 256"}]} },
      { type: 'check', question: {"type":"tf","q":"RT-2 produces robot actions as text tokens, using the same next-token prediction as for words.","answer":true,"explain":"Each action dimension is discretized into 256 bins whose tokens the model predicts, which let it be trained on robot data and web data together."} },
      { type: 'text', title: "Data, simulation and the sim-to-real gap", md: `Data is the bottleneck of robot learning, and every source has a catch:

- **Teleoperation**: people control the robot through a joystick, a VR headset or a matching "leader" arm. High quality, but slow and costly.
- **Pooled datasets**: the Open X-Embodiment collaboration (2023) combined over a million real trajectories from 22 kinds of robots, and training one model across many robots (*cross-embodiment* training) improved results on several of them.
- **Simulation**: unlimited, safe and parallel, but simulated physics, friction, contacts and images are never exactly real. Policies that work in simulation can fail on the real robot: the **sim-to-real gap**. **Domain randomization** narrows it by randomizing physics parameters, textures, lighting and camera positions during training, so that the real world looks like just another variation.
- **Human videos**: plentiful, but they show human hands rather than robot grippers, and have no action labels.
- **Generated data**: world models and video generators can produce training scenarios (see [World Models](#/concept/world-models)), with the same caveats about physics.` },
      { type: 'text', title: "Evaluating and deploying robot policies safely", md: `- **Many trials, with confidence intervals.** Physical trials are slow, so evaluations are small, and small evaluations are noisy: 20 trials with 14 successes put the true success rate somewhere between about 48% and 86% (95% Wilson interval; see [Statistical Inference](#/concept/stat-inference)). Compare policies on the same starting conditions, and report the number of trials.
- **Test generalization explicitly:** new objects, new scenes, new lighting, new instructions, and disturbances during the task.
- **Simulation benchmarks** help for fast iteration and reproducibility, but must be confirmed on real hardware.
- **Safety by design:** limits on force, torque and speed (especially near people), emergency stops, geofences around the workspace, monitoring that stops the robot when its inputs look unfamiliar, and human oversight, especially for learned policies whose failures are hard to predict.` }
    ],
    when: `- Imitation learning: an expert can demonstrate the task (teleoperation), and the task is hard to specify with a reward.
- Reinforcement learning in simulation: rewards are easy to define and a good simulator exists (locomotion, in-hand manipulation).
- Vision-language-action models: many tasks and objects, instructions in natural language, and enough robot data to fine-tune.
- Classical planning and control: the environment is structured and known (factory automation), and guarantees matter.`,
    pitfalls: `- Training only on perfect demonstrations: the policy never learns to recover from its own small mistakes.
- Averaging multimodal demonstrations (left or right around an obstacle) into one invalid action.
- Reporting success rates from a handful of trials: 20 trials leave a confidence interval about ±20 points wide.
- Deploying without force and speed limits, emergency stops and human oversight.`,
    takeaways: [
      'Robot learning is limited by data: there is no internet of robot actions, so demonstrations, simulation and cross-robot datasets matter.',
      'Behaviour cloning drifts because small errors lead to states the expert never showed; DAgger adds expert labels for the states the learner actually visits.',
      'Averaging demonstrations of different strategies gives an action no expert would take; generative policies (diffusion, flow matching) and action chunks avoid it.',
      'Vision-language-action models fine-tune a vision-language model to output actions, as tokens (RT-2, OpenVLA) or through an action expert (π0).',
      'Evaluate over many trials with confidence intervals, test generalization to new objects and scenes, and build in safety limits.'
    ],
    quiz: [
      {"type":"mcq","q":"What is covariate shift in behaviour cloning?","options":["The expert changes its mind","The learner's own small mistakes lead it to states that the demonstrations never covered","The robot's sensors drift over time","The training loss increases"],"answer":1,"explain":"The states the policy visits depend on the policy itself, so they drift away from the expert's states, where the policy was never trained."},
      {"type":"mcq","q":"What does DAgger add to the training data?","options":["More demonstrations of the same expert trajectories","Expert labels for the states that the learned policy actually visits","Random actions","Simulated images"],"answer":1,"explain":"Asking the expert what it would have done in the learner's states teaches the policy to recover from its own mistakes."},
      {"type":"tf","q":"When half the demonstrations go left around an obstacle and half go right, a policy trained with squared error will usually pick one of the two.","answer":false,"explain":"Squared error is minimized by the average action, here roughly straight ahead. Generative policies, such as diffusion policies, sample one of the modes instead."},
      {"type":"mcq","q":"What is the main reason for action chunking?","options":["To compress the model","Predicting several future actions at once means fewer decisions where errors can creep in, and smoother motion","To make the robot slower","To avoid using cameras"],"answer":1,"explain":"Executing a chunk of k actions per decision reduces compounding errors and captures multi-step motions."},
      {"type":"mcq","q":"What does domain randomization do?","options":["Randomizes the robot's goals","Varies physics, textures, lighting and camera positions in simulation so that the real world looks like one more variation","Mixes data from different robots","Shuffles the training data"],"answer":1,"explain":"Training across many simulated variations makes the policy robust to the differences between simulation and reality."},
      {"type":"fill","q":"A policy succeeds in 14 of 20 real-world trials. What is its measured success rate, in percent?","answer":["70","70%"],"explain":"14 / 20 = 70%, but with only 20 trials the 95% confidence interval runs from about 48% to 86%."}
    ],
    problems: [
      {
        id: 'action-tokens', title: 'Actions as Tokens', difficulty: 'easy',
        prompt: `Vision-language-action models such as RT-2 write each continuous action dimension as one of \`bins\` discrete tokens. Implement \`to_tokens(actions, low, high, bins=256)\` that returns a list of integers:

- clip each action to [low, high];
- map it linearly so that \`low\` → 0 and \`high\` → bins − 1;
- round to the nearest integer with Python's \`round\`.

\`\`\`text
to_tokens([-1, 1, 0.5], -1, 1)   ->  [0, 255, 191]       0.5 sits at 0.75 of the range: 0.75 × 255 = 191.25
to_tokens([2, -3], -1, 1)        ->  [255, 0]            clipped to the range first
\`\`\``,
        starter: `
def to_tokens(actions, low, high, bins=256):
    # TODO: clip, scale to 0 .. bins - 1, round
    pass
`,
        solution: `
def to_tokens(actions, low, high, bins=256):
    tokens = []
    for a in actions:
        a = min(max(a, low), high)
        tokens.append(int(round((a - low) / (high - low) * (bins - 1))))
    return tokens
`,
        hints: ['Clip with min(max(a, low), high).', 'The fraction (a − low) / (high − low) lies between 0 and 1.', 'Multiply by bins − 1 and round.'],
        explain: `With 256 bins over a range of 2, neighbouring tokens are 2 / 255 ≈ 0.0078 apart, so rounding moves an action by at most about 0.004, fine for most arm movements. The model can then predict actions with the same next-token machinery it uses for text.`,
        tests: [
          { call: 'to_tokens([-1, 1, 0.5], -1, 1)', expect: '[0, 255, 191]' },
          { name: 'clipping', call: 'to_tokens([2, -3], -1, 1)', expect: '[255, 0]' },
          { name: 'fewer bins', call: 'to_tokens([0.2, -0.9], -1, 1, bins=8)', expect: '[4, 0]' },
          { name: 'another range', call: 'to_tokens([0.0, 0.05, 0.1], 0.0, 0.1, bins=11)', expect: '[0, 5, 10]', hidden: true },
          { name: 'random actions vs reference', setup: 'import random\nrandom.seed(3)\nacts = [random.uniform(-1.5, 1.5) for _ in range(50)]', call: 'to_tokens(acts, -1, 1)', hidden: true }
        ]
      },
      {
        id: 'action-detokens', title: 'Tokens Back to Actions', difficulty: 'easy',
        prompt: `The robot needs continuous commands, so the predicted tokens are turned back into actions. Implement \`from_tokens(tokens, low, high, bins=256)\` that maps token 0 → \`low\`, token bins − 1 → \`high\`, and everything in between linearly. Return a list of floats.

\`\`\`text
from_tokens([0, 255], -1, 1)   ->  [-1.0, 1.0]
from_tokens([51], -1, 1)       ->  [-0.6]
\`\`\``,
        starter: `
def from_tokens(tokens, low, high, bins=256):
    # TODO
    pass
`,
        solution: `
def from_tokens(tokens, low, high, bins=256):
    return [low + t / (bins - 1) * (high - low) for t in tokens]
`,
        hints: ['Token t is the fraction t / (bins − 1) of the way from low to high.'],
        explain: `Encoding and decoding together lose at most half a bin: (high − low) / (2 (bins − 1)). More bins give finer control but a larger output vocabulary; action experts that output continuous values (flow matching in π0) avoid the trade-off.`,
        tests: [
          { call: 'from_tokens([0, 255], -1, 1)', expect: '[-1.0, 1.0]' },
          { name: 'a middle token', call: '[round(v, 6) for v in from_tokens([51], -1, 1)]', expect: '[-0.6]' },
          { name: 'fewer bins', call: 'from_tokens([0, 2, 4], 0, 1, bins=5)', expect: '[0.0, 0.5, 1.0]' },
          { name: 'round trip loses at most half a bin', setup: 'import random\nrandom.seed(5)\nacts = [random.uniform(-1, 1) for _ in range(200)]\ndef enc(a, bins=256):\n    return [int(round((x + 1) / 2 * (bins - 1))) for x in a]', call: 'max(abs(x - y) for x, y in zip(acts, from_tokens(enc(acts), -1, 1))) <= 1 / 255 + 1e-12', expect: 'True', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Ross, Gordon & Bagnell, A Reduction of Imitation Learning to No-Regret Online Learning (DAgger, 2011)', url: 'https://arxiv.org/abs/1011.0686' },
      { title: 'Brohan et al., RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023)', url: 'https://arxiv.org/abs/2307.15818' },
      { title: 'Kim et al., OpenVLA: An Open-Source Vision-Language-Action Model (2024)', url: 'https://arxiv.org/abs/2406.09246' },
      { title: 'Black et al., π0: A Vision-Language-Action Flow Model for General Robot Control (2024)', url: 'https://arxiv.org/abs/2410.24164' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
