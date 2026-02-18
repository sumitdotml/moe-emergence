# Phase 4 Training Fix Log

- **Date:** 2026-02-18
- **Commit reference:** `5026b2c3db2eefc06b62feb61f1dffd333aa82ec` (HEAD before committing this fix set)
- **Scope:** `moe_emergence/train.py`, `moe_emergence/tracking.py`

## Issues Fixed

1. Per-domain eval losses were computed from a mixed-batch scalar loss, not per-sequence domain losses.
2. `tracking.log_eval()` recomputed perplexity from aggregate eval loss, overriding the LM-based perplexity from `run_eval()`.
3. Training could stall indefinitely when datasets produced zero blocks.
4. CUDA data path lacked tuned defaults for pinned memory/workers/non-blocking copies.
5. Per-domain training losses were not logged.
6. Resume on MPS failed during RNG restoration (`torch.random.set_rng_state` expected CPU byte tensor).

## Root Causes

1. Eval aggregation used `outputs.loss.item()` for all domains in a mixed batch.
2. Tracking API accepted only eval loss and derived perplexity internally.
3. No explicit dataset length guards before infinite loader creation.
4. DataLoader/Tensor transfer settings were device-agnostic defaults.
5. `log_step()` supported domain losses but training loop never provided them.
6. RNG state loaded via `map_location=device` could become non-CPU tensor on MPS.

## Fixes Applied

1. Added `compute_sequence_lm_losses()` and switched eval domain metrics to per-sequence CE grouped by domain.
2. Updated `tracking.log_eval()` to accept `eval_lm_loss` and `eval_perplexity` explicitly.
3. Added hard fail-fast checks for zero-length train/eval datasets with actionable messages.
4. Added CUDA-only defaults: `num_workers=2`, `pin_memory=True`, `persistent_workers=True`, `non_blocking` tensor transfer.
5. Added per-domain training metric aggregation per optimizer step (`train/loss_code|math|prose`) with detached pre-backward computation.
6. Normalized RNG state tensors to CPU `uint8` before restoration in `load_checkpoint()`.

## Verification

1. `uv run ruff check moe_emergence/train.py moe_emergence/tracking.py`
2. Dense shakedown:
   - `uv run python -m moe_emergence.train --preset shakedown --run-name final-dense --max-steps 2 --eval-every 1 --save-every 1 --wandb-offline`
3. MoE shakedown:
   - `uv run python -m moe_emergence.train --preset shakedown --run-name final-moe --moe-layers 8 9 10 11 --max-steps 2 --eval-every 1 --save-every 1 --wandb-offline`
4. Resume test (MoE):
   - `uv run python -m moe_emergence.train --preset shakedown --run-name final-moe --moe-layers 8 9 10 11 --resume checkpoints/final-moe/ckpt-step-1.pt --max-steps 3 --eval-every 1 --save-every 1 --wandb-offline`
5. Per-domain eval distinctness check:
   - Dense eval step showed distinct values: `eval/loss_code=1.7937`, `eval/loss_math=3.4939`, `eval/loss_prose=3.7390`
   - MoE eval step showed distinct values: `eval/loss_code=1.8236`, `eval/loss_math=3.4686`, `eval/loss_prose=3.7365`

## Notes

1. Best-model selection remains on aggregate `eval/loss` (intentional design choice).
2. Checkpoint/resume behavior now succeeds on MPS in addition to CUDA/CPU paths.
