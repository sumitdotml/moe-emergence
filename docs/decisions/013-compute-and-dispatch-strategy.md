# Decision: Compute Strategy — Sequential Dispatch, GPU Utilization, and Scaling Boundaries

**Date:** 2026-02-08
**Status:** Accepted
**Context Commit:** `c66232d`

---

## Context

Phase 4 budgeted training runs will target NVIDIA GPUs; development and testing also support MPS (Apple Silicon) via device-agnostic code. The MoE implementation uses a Python `for` loop to dispatch tokens to experts sequentially, which differs from production MoE systems that use fused kernels. This decision documents: (a) why sequential dispatch is acceptable at our scale, (b) GPU utilization considerations for training, (c) batch size and gradient accumulation rationale, and (d) what would need to change at larger scale.

This serves as a comprehensive reference for understanding the compute characteristics of the project.

---

## 1. Expert Dispatch: Sequential Loop vs. Fused Kernels

### What the code does

Both `moe.py` (standalone) and `gpt2_moe.py` (GPT-2 wrapper) use the same dispatch pattern:

```python
for i, expert in enumerate(self.experts):                    # 8 iterations
    token_idx, topk_idx = torch.where(topk_indices == i)     # find tokens for expert i
    if len(token_idx) == 0:
        continue
    expert_input = hidden_flat[token_idx]                     # gather tokens
    expert_output = expert(expert_input)                      # run MLP on GPU
    results[token_idx] += weights[...] * expert_output        # scatter-add results
```

This loops over 8 experts, issuing separate GPU kernel launches for each. The pattern comes from Mistral's reference implementation (`mistral-inference/moe.py`).

### What each iteration does on the GPU

The following cost estimates are **order-of-magnitude approximations** based on typical CUDA kernel launch overhead and small-GEMM latency ranges from PyTorch profiling literature. They have not been measured on this specific codebase. Actual values will vary by GPU model, driver version, and batch contents.

| Operation | GPU Work | Estimated Cost (order-of-magnitude) |
|-----------|----------|-------------------------------------|
| `torch.where(topk_indices == i)` | Element-wise comparison + nonzero | ~5-10μs |
| `hidden_flat[token_idx]` (gather) | Index select into contiguous buffer | ~5-10μs |
| `expert(expert_input)` — Linear(768→3072) | GEMM kernel | ~20-50μs |
| `expert(expert_input)` — GELU activation | Element-wise | ~5μs |
| `expert(expert_input)` — Linear(3072→768) | GEMM kernel | ~20-50μs |
| `weights * expert_output` (scale) | Element-wise multiply | ~5μs |
| `results[token_idx] += ...` (scatter-add) | Indexed addition | ~5-10μs |

**Estimated total per expert:** ~65-140μs (not profiled)
**Estimated total for 8 experts:** ~0.5-1.1ms
**Estimated total for 4 MoE layers per step:** ~2-4.5ms

### Why this is acceptable at our scale

A full GPT-2 small forward + backward pass at batch_size=2, seq_len=512 is estimated at **~100-300ms** on a typical NVIDIA GPU (this range is a rough estimate, not profiled on our specific setup). The sequential dispatch overhead of ~2-4.5ms would represent **~1-4% of total step time** — the key point is that it's a small fraction regardless of exact values.

Key numbers for our configuration:
- **Tokens per batch:** 2 × 512 = 1,024
- **Tokens per expert (top-1, balanced):** ~128
- **Expert hidden dim:** 768 → 3072 → 768
- **Matrix sizes per expert:** [128, 768] × [768, 3072] — these are small GEMMs that don't saturate GPU compute anyway

Even with perfect parallelization, the actual compute savings would be small because these matrix multiplies are memory-bandwidth-bound at this size, not compute-bound.

### What production systems do differently

Production MoE implementations (Megablocks, Tutel, DeepSpeed-MoE) replace the loop with fused dispatch:

**1. Permute-then-batch approach (Megablocks):**
```python
# Conceptual pseudocode — not our code
sorted_tokens = permute_tokens_by_expert(hidden_flat, topk_indices)
expert_boundaries = compute_boundaries(topk_indices, n_experts)
# Single grouped GEMM: processes all experts in one kernel launch
all_outputs = grouped_gemm(sorted_tokens, expert_weights, expert_boundaries)
results = unpermute_tokens(all_outputs, original_order)
```

This turns 8 kernel launches into 1 by sorting tokens by expert assignment and running a grouped/batched matrix multiply. The GPU sees one large operation instead of 8 small ones.

**2. Capacity factor with token dropping (Switch Transformer):**
```python
# Each expert has a fixed buffer size = (tokens / n_experts) * capacity_factor
# Overflow tokens are dropped; underflow tokens are padded
# Enables fixed-shape batched matmul across all experts
```

**3. Triton custom kernels:**
Hand-written kernels that fuse gather → expert forward → scatter into a single GPU launch, eliminating all Python-level loop overhead and intermediate memory allocations.

### When the loop becomes a bottleneck

The sequential dispatch becomes problematic when:
- **n_experts > 32**: Loop overhead grows linearly; 64 experts = 64 kernel launches
- **Tokens per batch > 100K**: Gather/scatter operations become expensive at large tensor sizes
- **Multi-GPU training**: The loop prevents overlapping expert compute across devices (expert parallelism requires sorted dispatch)
- **Top-k > 1 at large scale**: Each token appears in multiple expert queues, multiplying the scatter-add cost

**None of these apply to our project** (8 experts, ~1K tokens/batch, single GPU, top-1).

---

## 2. GPU Utilization and Device Handling

### Automatic device placement

PyTorch handles CUDA dispatch transparently. The training script calls:
```python
model = model.to(device)        # moves all parameters to GPU
input_ids = batch["input_ids"].to(device)  # moves input tensors to GPU
```

After this, every operation in the forward/backward pass (including inside MoE expert loops) runs on the GPU automatically. No CUDA-specific code is needed beyond device placement.

### What runs on GPU vs. CPU

| Component | Device | Notes |
|-----------|--------|-------|
| All model parameters | GPU | Placed by `model.to(device)` |
| Router softmax + topk | GPU | Standard PyTorch ops |
| Expert forward passes | GPU | Linear layers run as cuBLAS GEMMs |
| Loss computation (LM, LB, Z) | GPU | Tensor operations on GPU tensors |
| Gradient computation | GPU | Autograd backward pass |
| Optimizer step (AdamW) | GPU | Parameter updates on device |
| Python `for` loop control flow | CPU | Only the loop iteration; all tensor ops inside are GPU |
| `torch.where()` index computation | GPU | Returns GPU tensors |
| DataLoader collation | CPU | Standard; tensors moved to GPU per batch |

### Mixed precision

We are **not** using explicit mixed precision (AMP/fp16/bf16) for this project.

**Why not:**
- GPT-2 small is 124M parameters — fits comfortably in fp32 on any modern GPU
- AMP adds complexity (loss scaling, grad scaler, potential numerical issues with router softmax)
- The budget savings from fp16 (~2x memory, ~1.5x throughput) are not necessary when the model already fits easily
- Router logits and softmax probabilities can be sensitive to precision — MoE papers note that router gradients benefit from fp32 stability

**When you would use mixed precision:**
- Model size approaches GPU memory limits (e.g., GPT-2 large/XL, or 32+ experts)
- Training throughput is the primary bottleneck and you need to maximize tokens/second
- Budget is tight enough that 1.5x speedup materially changes what's feasible
- In that case, keep router computation in fp32 (via `torch.autocast` exclusion) while using fp16/bf16 for expert forward passes

### MPS vs. CUDA parity

Budgeted training runs target NVIDIA CUDA GPUs for performance and reproducibility. MPS (Apple Silicon) is supported for local development and shakedown testing. Key differences:

| Aspect | CUDA | MPS |
|--------|------|-----|
| Batch size headroom | Likely supports batch_size=4-8 | Conservative batch_size=2 |
| float64 support | Full | Limited (some ops fall back to CPU) |
| Deterministic mode | `torch.use_deterministic_algorithms(True)` | Not all ops have deterministic MPS implementations |
| Mixed precision | Full AMP support | Limited bf16 support |
| Kernel launch overhead | ~5-10μs | ~10-30μs (higher than CUDA) |

For this project, the code is device-agnostic. The `--device auto` flag auto-detects CUDA > MPS > CPU.

---

## 3. Batch Size and Gradient Accumulation

### Chosen configuration

| Parameter | Value | Effective |
|-----------|-------|-----------|
| `--batch-size` | 2 | Micro-batch (per forward pass) |
| `--grad-accum-steps` | 4 | Accumulation steps |
| **Effective batch** | **8** | 2 × 4 = 8 sequences per optimizer step |

### Rationale

The V3 design spec specifies `batch_size=8`. The Phase 4 plan splits this into `2×4` for memory safety across devices:

**Memory estimation for GPT-2 small + 8 experts (top-1, 4 MoE layers):**
- Base GPT-2 parameters: ~124M × 4 bytes = ~496MB
- MoE expert parameters: 4 layers × 8 experts × 4.72M params × 4 bytes = ~604MB (each expert is a GPT-2 MLP: Linear(768→3072) + Linear(3072→768) + biases = 4,722,432 params). Note: experts replace the original MLP, so 4 layers × 1 original MLP (~75MB) is freed, net add ≈ ~529MB.
- Optimizer states (AdamW): 2× parameter memory = ~2.1GB
- Activations (batch_size=2, seq_len=512): ~200-400MB
- **Total estimated:** ~3.3-3.5GB

This fits comfortably on 8GB+ GPUs. Batch_size=8 without accumulation would multiply activation memory by 4× (~1.6GB activations), still fine on 16GB+ CUDA GPUs but tight on 8GB MPS.

### OOM fallback ladder

If training OOMs:
1. `batch_size=1, grad_accum_steps=8` — halves activation memory, same effective batch
2. `block_size=256` — halves sequence length, significantly reduces activation memory (attention activations scale quadratically with sequence length, so ~4× reduction for attention; MLP activations scale linearly, so ~2× reduction there; net effect depends on the balance but expect roughly ~2-3× total reduction)

### Why not larger batches?

Larger effective batches (16, 32) are possible on NVIDIA GPUs with sufficient VRAM but:
- This is a small-data regime (~12K training blocks total). Larger batches mean fewer optimizer steps per epoch.
- At `effective_batch=8` with `max_steps=10000`, the model sees each block ~6.7 times. Doubling batch size halves this to ~3.3 passes — potentially undertrained.
- The learning rate (`5e-5`) was chosen for effective_batch=8. Scaling batch size requires scaling LR (linear scaling rule), adding another variable.

---

## 4. Kernel Launch Overhead Analysis

### What "kernel launch overhead" means

Every PyTorch operation (matmul, softmax, element-wise add) dispatches a CUDA kernel to the GPU. Each dispatch has fixed overhead:
- **CPU-side:** Python → PyTorch dispatcher → CUDA driver: ~5-20μs
- **GPU-side:** Kernel scheduling + launch: ~2-5μs
- **Total per launch:** ~7-25μs

For small operations (tiny matrix multiplies, scalar reductions), this overhead can dominate actual compute time.

### Our overhead budget

Per training step (forward + backward):
- GPT-2 transformer blocks (8 non-MoE): ~hundreds of kernel launches (attention, LN, MLP) — this is the baseline
- MoE dispatch loop (4 layers × 8 experts): ~32 additional expert forward passes + ~32 gather/scatter ops ≈ **~128 extra kernel launches**
- Auxiliary loss computation: ~16 kernel launches (4 layers × 2 losses × ~2 ops each)

**Estimated overhead from MoE sequential dispatch:** ~128 × 15μs ≈ **~2ms per step** (order-of-magnitude estimate, not profiled)

For context: a full training step (forward + backward + optimizer) is estimated at ~200-500ms. The sequential dispatch would add roughly ~0.4-1% overhead.

### When overhead matters

| Scale | Tokens/batch | Experts | Loop overhead | Step time | Overhead % |
|-------|-------------|---------|---------------|-----------|------------|
| **Ours** | 1,024 | 8 | ~2ms | ~200-500ms | **~0.4-1%** |
| Medium | 32,768 | 16 | ~8ms | ~50-100ms | ~8-16% |
| Large | 1M+ | 64 | ~32ms | ~200ms | ~16% |
| Production | 4M+ | 128 | ~64ms | ~500ms | ~13% |

At medium scale and above, the loop overhead becomes significant enough to justify fused kernels.

---

## 5. What Would Change at Larger Scale

If this project were scaled up (larger model, more experts, multi-GPU), here's what would need to change, in priority order:

### Scale 1: Bigger model (GPT-2 Medium/Large, same 8 experts)

**Changes needed:**
- Mixed precision (AMP with fp16/bf16) to fit in memory
- Gradient checkpointing to trade compute for memory
- Possibly increase effective batch size and scale LR accordingly

**No dispatch changes needed** — with the same 8 experts but larger hidden dimensions, each expert GEMM does more work, so the fixed loop overhead becomes a smaller fraction of compute time. If expert count also increases to 32+, dispatch changes (Scale 2) become relevant — see the bottleneck thresholds in Section 1.

### Scale 2: High-throughput training (100K+ tokens/batch)

**Changes needed:**
- Replace Python expert loop with **Megablocks** grouped GEMM or **Tutel** dispatched MoE
- Use capacity factors to bound per-expert buffer sizes
- Consider token dropping for load balance hard caps

**Why:** At 100K tokens, each expert processes ~12.5K tokens (top-1, 8 experts). The gather/scatter operations become significant, and the loop prevents the GPU from overlapping expert computation.

### Scale 3: Multi-GPU training (expert parallelism)

**Changes needed:**
- Spread experts across GPUs (expert parallelism) using DeepSpeed-MoE or Fairscale
- All-to-all communication to route tokens between GPUs
- Sorted dispatch is mandatory (can't loop-dispatch across devices)

**Why:** With experts on different GPUs, you need to send tokens to the right GPU before processing. This requires a global sort → all-to-all → compute → all-to-all → unsort pipeline that is fundamentally incompatible with a Python loop.

### Scale 4: Production inference (Mixtral-scale)

**Changes needed:**
- Triton or CUDA custom kernels for fused gather-compute-scatter
- Expert offloading (some experts on CPU/disk, loaded on demand)
- Speculative expert prediction for latency hiding

---

## Decision

**Accept sequential expert dispatch for this project.** The estimated overhead is small (order of a few percent of step time) at our scale (8 experts, GPT-2 small, ~1K tokens/batch). Fused kernels would add implementation complexity with negligible throughput benefit.

**Accept fp32 training without mixed precision.** The model fits comfortably in GPU memory. Router stability benefits from fp32 precision.

**Accept batch_size=2 with grad_accum=4.** Conservative for cross-device compatibility, with documented OOM fallback ladder.

---

## Consequences

- **Positive:**
  - Simpler codebase — no Triton/CUDA dependencies, no Megablocks integration
  - Easier debugging — standard PyTorch profiling tools work without custom kernel complications
  - Cross-device portability — same code runs on CUDA, MPS, and CPU without conditional paths
  - Reproducibility — fp32 avoids mixed-precision nondeterminism

- **Negative:**
  - Small throughput penalty from sequential dispatch (estimated few percent, acceptable at this scale)
  - Dispatch loop overhead grows linearly with expert count; scaling to 32+ experts or 100K+ tokens/batch would require fused dispatch (see Section 5)
  - No multi-GPU support without rewriting the dispatch layer

- **Risks:**
  - If training is unexpectedly slow, profile before assuming dispatch is the bottleneck — the issue is more likely DataLoader I/O, data loading, or W&B logging overhead

---

## References

- Mistral reference dispatch: `mistral-inference/moe.py` — our pattern's source
- Megablocks: Gale et al. "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" (2022) — grouped GEMM approach
- Switch Transformer: Fedus et al. (2021) — capacity factors and token dropping
- Tutel: Hwang et al. "Tutel: Adaptive Mixture-of-Experts at Scale" (2023) — adaptive dispatch
- DeepSpeed-MoE: Rajbhandari et al. (2022) — multi-GPU expert parallelism
- ST-MoE: Zoph et al. (2022) — router precision considerations
- README.md "Known Bottlenecks" section — existing project documentation
- Phase 4 Training Plan "Out of Scope" — dispatch-kernel optimization explicitly deferred
- Cross-model audit debate 008, Finding #7 — sequential dispatch rated P2 (acceptable)
