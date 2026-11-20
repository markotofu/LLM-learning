"""Chart data for batch L5 (training at scale, SSMs, multimodal). Apply AFTER patches/l5-text.txt."""
import json, sys, math
r = lambda v, d=4: [None if x is None else round(float(x), d) for x in v]
out = []
def chart(concept, where, title, spec, md="", after=""):
    pos = " ".join(f'{k}="{v}"' for k, v in where.items())
    block = [f"### INSERT concept={concept} {pos}", f"@@ chart | {title}"] + ([md] if md else []) + ["%% spec", json.dumps(spec, ensure_ascii=False)]
    if after: block += ["%% after", after]
    out.append("\n".join(block))

# ---------------- training at scale: memory per GPU under ZeRO stages (7B model, 8 GPUs)
N, G = 6.74e9, 8
stages = [("plain data parallel", 2 + 2 + 12), ("ZeRO-1: shard optimizer states", 2 + 2 + 12 / G), ("ZeRO-2: also shard gradients", 2 + (2 + 12) / G), ("ZeRO-3 / FSDP: shard everything", (2 + 2 + 12) / G)]
vals = [b * N / 1e9 for _, b in stages]
chart("training-at-scale", {"after": "Data parallelism and sharding"}, "Memory per GPU for a 7B model on 8 GPUs",
      {"kind": "bar", "x": ["data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"], "series": [{"name": "weights + gradients + Adam states per GPU (GB, before activations)", "y": r(vals, 1)}], "xLabel": "strategy", "yLabel": "GB per GPU", "yMin": 0},
      md="Mixed-precision Adam: 2 bytes of bf16 weights, 2 of gradients and 12 of fp32 optimizer state (master weights and two moments) per parameter:",
      after=f"Plain data parallelism keeps a full copy of all 16 bytes per parameter on every GPU: about {vals[0]:.0f} GB, more than an 80 GB GPU holds. Sharding the optimizer states across the 8 GPUs (ZeRO-1) brings that to {vals[1]:.0f} GB, sharding the gradients too (ZeRO-2) to {vals[2]:.0f} GB, and sharding the weights as well (ZeRO-3, PyTorch's FSDP) to {vals[3]:.1f} GB, at the price of gathering each layer's weights over the network just before it runs.")

# ---------------- SSMs: memory against context (the lesson's settings)
layers, kv_heads, head_dim, d_model, state_dim, nbytes = 32, 8, 128, 4096, 16, 2
kv_per_token = 2 * layers * kv_heads * head_dim * nbytes; ssm_state = layers * d_model * state_dim * nbytes
ctxs = [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000]
gib = lambda b: b / 2 ** 30
chart("ssm-architectures", {"after": "Memory as context grows: KV cache versus a fixed state"}, "Memory for one sequence as the context grows",
      {"kind": "line", "x": ctxs, "xLog": True, "yLog": True, "series": [
          {"name": "attention: KV cache", "y": r([gib(c * kv_per_token) for c in ctxs], 4)},
          {"name": "hybrid, 1 attention layer in 8", "y": r([gib(c * kv_per_token / 8 + ssm_state * 7 / 8) for c in ctxs], 4)},
          {"name": "state-space model: fixed state", "y": r([gib(ssm_state) for c in ctxs], 4)}],
       "xLabel": "context length in tokens (log scale)", "yLabel": "GiB (log scale)"},
      after="The KV cache grows in a straight line with the context (on log axes, a diagonal), passing 100 GiB around a million tokens. The state-space model's memory is a flat line: its state has the same size after ten tokens or ten million. The hybrid pays for its few attention layers, which still grow, but sits an order of magnitude below pure attention.")

# ---------------- multimodal: image tokens against resolution
def image_tokens(w, h, patch=14, pool=1): return math.ceil(math.ceil(w / patch) / pool) * math.ceil(math.ceil(h / patch) / pool)
def tiled(w, h, tile=336, patch=14, pool=1): return (math.ceil(w / tile) * math.ceil(h / tile) + 1) * image_tokens(tile, tile, patch, pool)
sides = [224, 336, 448, 672, 896, 1344]
chart("multimodal-models", {"after": "An image-token budget calculator"}, "Image tokens for a square image",
      {"kind": "line", "x": sides, "xLog": True, "yLog": True, "series": [
          {"name": "whole image at full resolution, 14-pixel patches", "y": [image_tokens(s, s) for s in sides]},
          {"name": "336-pixel tiles plus a thumbnail", "y": [tiled(s, s) for s in sides]},
          {"name": "whole image with 2 × 2 token pooling", "y": [image_tokens(s, s, pool=2) for s in sides]}],
       "xLabel": "image side in pixels (log scale)", "yLabel": "tokens (log scale)"},
      after="Tokens grow with the square of the side: doubling the resolution quadruples the cost. A 1,344-pixel square photo costs over 9,000 tokens at full resolution, the length of a short book chapter. Tiling adds a thumbnail but keeps each tile at the encoder's native size, and pooling neighbouring tokens divides the count by four, the usual compromise between seeing small details and staying within the context budget.")

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(out) + "\n")
print(len(out), "charts", [round(v, 1) for v in vals], image_tokens(1344, 1344), tiled(1344, 1344), image_tokens(1344, 1344, pool=2))
