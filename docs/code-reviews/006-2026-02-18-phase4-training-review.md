# Code Review: Phase 4 Training Readiness

- **Date:** 2026-02-18
- **Commit reviewed:** `5026b2c3db2eefc06b62feb61f1dffd333aa82ec`
- **Scope:** `moe_emergence/train.py`, `moe_emergence/data.py`, `moe_emergence/moe.py`, `moe_emergence/gpt2_moe.py`, `moe_emergence/tracking.py`, `moe_emergence/gpt2_inference.py`

## Findings

### HIGH

1. **Per-domain eval metrics are not actually per-domain.**
   - **Where:** `moe_emergence/train.py:369`, `moe_emergence/train.py:371`, `moe_emergence/train.py:385`, `moe_emergence/train.py:389`
   - **What:** A single batch-level LM loss (`outputs.loss`) is appended once per sample-domain label in the batch, so all domains in a mixed batch receive the same value.
   - **Impact:** `eval/loss_code`, `eval/loss_math`, `eval/loss_prose` and derived per-domain perplexities are biased and can mislead specialization analysis.
   - **Fix direction:** Compute per-example/token losses (`reduction="none"`), then aggregate by domain mask rather than copying one mixed-batch scalar.

### MEDIUM

2. **W&B eval perplexity uses `eval/loss` (LM + aux) instead of LM loss.**
   - **Where:** `moe_emergence/tracking.py:140`, `moe_emergence/tracking.py:142`, `moe_emergence/train.py:377`, `moe_emergence/train.py:782`
   - **What:** `tracking.log_eval()` logs `eval/perplexity = exp(eval_loss)` where `eval_loss` is total loss (includes LB/Z for MoE), not LM loss.
   - **Impact:** Dashboard perplexity is mathematically inconsistent with language modeling perplexity and inconsistent with `run_eval()` output.
   - **Fix direction:** Pass `eval/lm_loss` into `log_eval` and compute perplexity from LM loss only.

3. **Potential silent hang if training dataset has zero blocks.**
   - **Where:** `moe_emergence/train.py:311`, `moe_emergence/train.py:313`, `moe_emergence/train.py:516`
   - **What:** `infinite_loader()` loops forever; if `DataLoader` is empty (small `--size-mb`, large `--block-size`, or data filtering), generator never yields and training appears stuck.
   - **Impact:** Hard-to-diagnose stalled runs.
   - **Fix direction:** Validate `len(train_dataset) > 0` and `len(eval_dataset) > 0` before creating infinite iterator; fail fast with clear error.

### LOW

4. **CUDA input pipeline is functional but not tuned.**
   - **Where:** `moe_emergence/train.py:537`, `moe_emergence/train.py:543`, `moe_emergence/train.py:641`
   - **What:** `DataLoader` uses default worker/pinning settings and `.to(device)` is blocking.
   - **Impact:** May cap tokens/sec on remote GPUs.
   - **Fix direction:** For CUDA runs, consider `pin_memory=True`, `num_workers>0`, and `.to(device, non_blocking=True)`.

## Verified Correct Against V3/Phase-4 Plan

1. MoE integration is drop-in (`GPT2MLP` replacement) without rewriting HuggingFace block forward.
2. Warm-start uses `copy.deepcopy(original_mlp)` with small expert perturbation.
3. Router top-1 uses STE to preserve forward scale while enabling backward signal.
4. Load-balancing and z-loss are computed per MoE layer and averaged.
5. `no-lb` preset correctly sets `lb_coef=0.0` and `noise_std=0.0`.
6. Resume checkpoints include optimizer/scheduler and RNG states; model-only snapshots use `.safetensors` + JSON sidecar.
7. Train/eval split is text-level pre-packing, which avoids document leakage.

## Suggested Next Steps

1. Fix per-domain eval computation and W&B perplexity before budgeted runs.
2. Add explicit empty-dataset guards to prevent infinite-loader stalls.
3. Run mandatory dense + MoE shakedown after fixes, then proceed to budgeted run order.
