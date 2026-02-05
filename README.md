# MoE Emergence

## Objective

To train a small MoE model on 3 distinct domains (code, math, natural language), and create visualizations / consumable artifacts showing:

- Expert specialization emergence - which experts become domain specialists
- The load balancing problem - what happens without auxiliary loss
- Routing patterns - heatmaps of which tokens → which experts

## Progress Log

### Phase 1: MoE Components [DONE]

- [x] Router (NoisyTop-k, STE for top-1, noise annealing)
- [x] Load balancing loss (Switch Transformer style)
- [x] Z-loss (router logit stabilization)
- [x] SwiGLU FFN reference impl (standalone, was thinking of using it but I'm using pretrained GPT-2 as base model, so I'll have to stick to GELU)

### Phase 2: GPT-2 Integration [DONE]

- [x] `MoEWrapper` — drop-in replacement for `GPT2MLP`
- [x] `install_moe_layers()` — surgery function for layers 8-11
- [x] Warm-start via `deepcopy(original_mlp)` + tiny noise
- [x] `collect_aux_outputs()` — retrieves routing stats for loss computation
- [x] Smoke tests pass (shape, gradient flow, losses) when running `gpt2_moe.py`
- [x] Full integration verification with actual GPT-2 model (10/10 tests passed)
  - See: `docs/experiments/run-001-gpt2-integration-verification.md`

### Phase 3: Dataset Preparation

- [x] Sequence packing (not padding)
- [x] `PackedMixedDomainDataset` class
- [x] Train/eval split at TEXT level (before packing) — see decision 008
- [x] Dataset choices finalized — see decision 010:
  - **Code**: CodeParrot-clean (diverse Python, multiple licenses)
  - **Math**: MathQA from allenai (29K word problems with rationales)
  - **Prose**: AllenAI C4 English (natural web text, well-filtered)
- [x] Shuffle buffer rationale verified — not needed, see decision 006
- [x] Token balancing required — 1.9x imbalance without `--balance-tokens`, see decision 005
- [x] Shuffle-before-truncate — see decision 012
- [x] W&B experiment tracking — see decision 009, `moe_emergence/tracking.py`

### Phase 4: Training Infrastructure

- [ ] Training loop with LM + LB + Z losses
- [ ] Logging & checkpointing
- [ ] Dense baseline run

### Phase 5: Experiments

- [ ] MoE main run (8 experts, top-1, LB on)
- [ ] Ablation: no load balancing (expect collapse)
- [ ] Ablation: top-2 routing (optional, if budget allows)

### Phase 6: Analysis

- [ ] Expert specialization heatmaps
- [ ] Router entropy over training
- [ ] Fine-grained token-type routing

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

| Component            | Status | File                         | Notes                                        |
| -------------------- | ------ | ---------------------------- | -------------------------------------------- |
| Router               | ✅     | `moe.py`                     | NoisyTop-k, STE for top-1, noise annealing   |
| MoE Layer            | ✅     | `moe.py`                     | Dispatches tokens to experts                 |
| Load Balancing Loss  | ✅     | `moe.py`                     | Switch Transformer style (exported function) |
| Z-Loss               | ✅     | `moe.py`                     | Router logit stabilization (exported)        |
| MoE Wrapper          | ✅     | `gpt2_moe.py`                | Drop-in replacement for `GPT2MLP`            |
| GPT-2 Surgery        | ✅     | `gpt2_moe.py`                | `install_moe_layers()` function              |
| Aux Collection       | ✅     | `gpt2_moe.py`                | `collect_aux_outputs()` with clean probs     |
| Verification         | ✅     | `verify_gpt2_integration.py` | 10 comprehensive tests                       |
| Inference Playground | ✅     | `gpt2_inference.py`          | Supports vanilla/MoE/checkpoints             |
| Sequence Packing     | ✅     | `data.py`                    | Efficient dataset without padding            |
| Training Loop        | ⬜     | `train.py`                   | With aux loss collection                     |
| Collapse Detection   | ⬜     | —                            | Early stopping for ablation                  |
| Visualization        | ⬜     | —                            | Heatmaps, entropy plots                      |

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

## Quick Start

### Setup

```bash
uv sync                  # install dependencies
uv pip install -e .      # install package (enables imports)
```

### Inference Playground

Can play with GPT-2 generation (vanilla or MoE):

```bash
# vanilla GPT-2
uv run python moe_emergence/gpt2_inference.py --prompt "Once upon a time"

# untrained MoE (Phase 2)
uv run python moe_emergence/gpt2_inference.py --moe --prompt "def fibonacci(n):"

# trained MoE from checkpoint (Phase 5+) (btw this isn't ready yet since I haven't done MoE training yet, coming soon)
uv run python moe_emergence/gpt2_inference.py \
  --checkpoint checkpoints/run-002-step-10000.pt \
  --prompt "Solve the equation x^2 + 5x + 6 = 0"

# creative sampling
uv run python moe_emergence/gpt2_inference.py \
  --prompt "In a distant galaxy" \
  --sample --temperature 0.9 --max-tokens 100
```

See `python moe_emergence/gpt2_inference.py --help` for all options.

### Verification

Run Phase 2 verification (10 comprehensive tests):

```bash
uv run python moe_emergence/verify_gpt2_integration.py
```

## Code Formatting

Ruff being used for formatting here.

- **VSCode / Cursor**: Install the Ruff extension. Settings are in `.vscode/settings.json`.
- **Neovim**: Configure Ruff inside nvim itself. For reference, see how I've done [mine](https://github.com/search?q=repo%3Asumitdotml%2Fdotfiles%20ruff&type=code). For formatting, my command is `<leader>gf`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

For third-party software, datasets, and academic references, see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
