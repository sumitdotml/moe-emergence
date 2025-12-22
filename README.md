# MoE Emergence

## Objective

To train a small MoE model on 3 distinct domains (code, math, natural language), and create visualizations / consumable artifacts showing:

- Expert specialization emergence - which experts become domain specialists
- The load balancing problem - what happens without auxiliary loss
- Routing patterns - heatmaps of which tokens → which experts

## Progress Log

**MoE Components**

- [x] Router (load balancing, z-loss, noisy routing, STE for top-1)
- [x] MoE Layer wrapper
- [x] SwiGLU FFN (was thinking of using it, but since I'm using pretrained GPT-2 as base model, I'll use GELU)

**GPT-2 Integration**

- [ ] Install MoE wrapper into pretrained GPT-2 (layers 8-11)
- [ ] Warm-start experts via `deepcopy(original_mlp)`

**Data & Training**

- [ ] Dataset preparation (code/math/prose with sequence packing)
- [ ] Training loop with aux losses

**Experiments**

- [ ] Dense baseline run
- [ ] MoE main run (8 experts, top-1, load balancing on)
- [ ] Ablation: no load balancing (expect collapse)
- [ ] Ablation: top-2 routing (optional, budget permitting)

**Analysis**

- [ ] Expert specialization heatmaps (experts × domains)
- [ ] Router entropy over training
- [ ] Fine-grained token-type routing (Python keywords vs identifiers, etc.)

## Code to Write

**Base Architecture**

I could implement the GPT-2 architecture from scratch, but since I need to exactly match the architecture to load the pretrained weights, it's deffo better to use the provided stack of modules from HuggingFace to save the unnecessary friction. Using `GPT2LMHeadModel.from_pretrained('gpt2')`. Tbh I'd be a bit bothered normally if I simply imported huge chunks of architecture modules, but I've written the GPT-2 architecture too many times already, so... it's aight

| Component           | HuggingFace Class          | What it does                |
| ------------------- | -------------------------- | --------------------------- |
| Token Embeddings    | `nn.Embedding(50257, 768)` | Vocab → vectors             |
| Position Embeddings | `nn.Embedding(1024, 768)`  | Learned absolute positions  |
| Self-Attention      | `GPT2Attention`            | 12-head MHA, 768 dim        |
| FFN / MLP           | `GPT2MLP`                  | GELU, 768 → 3072 → 768      |
| LayerNorm           | Pre-norm                   | Before attention & MLP      |
| Transformer Block   | `GPT2Block`                | Attention + MLP + residuals |

Source: [modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

**What I Need to Write**

| Component           | Status | File     | Notes                                   |
| ------------------- | ------ | -------- | --------------------------------------- |
| Router              | ✅     | `moe.py` | With noisy routing, STE, load balancing |
| MoE Layer           | ✅     | `moe.py` | Dispatches tokens to experts            |
| Load Balancing Loss | ✅     | `moe.py` | Switch Transformer style                |
| Z-Loss              | ✅     | `moe.py` | Router logit stabilization              |
| MoE Wrapper         | ⬜     | —        | Drop-in replacement for `GPT2MLP`       |
| GPT-2 Surgery       | ⬜     | —        | `block.mlp = MoEWrapper(...)`           |
| Sequence Packing    | ⬜     | —        | Efficient dataset without padding       |
| Training Loop       | ⬜     | —        | With aux loss collection                |
| Collapse Detection  | ⬜     | —        | Early stopping for ablation             |
| Visualization       | ⬜     | —        | Heatmaps, entropy plots                 |

## Known Bottlenecks

**Sequential Expert Dispatch**

The current implementation processes experts one at a time in a Python loop:

```python
for i, expert in enumerate(self.experts):
    token_idx, topk_idx = torch.where(topk_indices == i)
    results[token_idx] += weights[...] * expert(x_flat[token_idx])
```

This is fine for learning and small-scale experiments like mine (8 experts, ~50M params each), but doesn't parallelize expert computation. Production MoE systems use fused kernels (e.g., Triton, Megablocks) or grouped matrix multiplications to dispatch all experts in parallel.

**Memory: All Experts Loaded**

All expert weights live in GPU memory even though only top-k are active per token. For 8 experts with top-2 routing, 75% of expert parameters are "idle" at any given moment. Techniques like expert parallelism (spreading experts across GPUs) or expert offloading can help at scale, but add complexity.

At my scale (8 experts, ~50M params each), this isn't a real bottleneck & everything fits comfortably in GPU memory. Just adding this note to self that this becomes a genuine problem at 64+ experts or multi-billion parameter experts, where we'd need distributed solutions like DeepSpeed-MoE or Megablocks.

**Load Imbalance**

Even with auxiliary load balancing loss, real-world data distributions can still cause some experts to be busier than others. The aux loss encourages balance but doesn't guarantee it. Capacity factors and token dropping are used in some implementations to hard-cap expert load. For this project, I'll yolo it with just the aux loss & not overengineer it.

## Code Formatting

Ruff being used for formatting here.

- **VSCode / Cursor**: Install the Ruff extension. Settings are in `.vscode/settings.json`.
- **Neovim**: Configure Ruff inside nvim itself. For reference, see how I've done [mine](https://github.com/search?q=repo%3Asumitdotml%2Fdotfiles%20ruff&type=code). For formatting, my command is `<leader>gf`.
