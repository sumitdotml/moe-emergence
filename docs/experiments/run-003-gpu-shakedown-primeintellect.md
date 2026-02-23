# Experiment: GPU Shakedown (PrimeIntellect)

**Run ID:** run-003
**Date:** 2026-02-23
**Status:** Completed
**Type:** Verification
**Phase:** 4
**Commit:** `199e632`
**Commands:**
```bash
uv run python -m moe_emergence.train --preset shakedown --run-name shake-dense --device cuda
uv run python -m moe_emergence.train --preset shakedown --run-name shake-moe --device cuda --moe-layers 8 9 10 11
```

---

## Objective

Validate that both dense and MoE training run correctly on cloud GPU (PrimeIntellect
RTX 4090) before committing to the budgeted multi-hour runs. This is the mandatory
shakedown gate from the Phase 4 training plan.

---

## Configuration

### Model (Dense)
```yaml
base_model: gpt2
mode: dense
```

### Model (MoE)
```yaml
base_model: gpt2
moe_layers: [8, 9, 10, 11]
num_experts: 8
top_k: 1
```

### Training (Both)
```yaml
preset: shakedown
max_steps: 100
batch_size: 2
grad_accum_steps: 4
effective_batch: 8
block_size: 512
learning_rate: 5e-5
warmup_fraction: 0.1
eval_every: 50
save_every: 50
data_size_mb: 1.0
```

### Loss Coefficients (MoE)
```yaml
lb_coef: 0.01
z_coef: 0.001
```

---

## Environment

- **Hardware:** 1x RTX 4090 24GB (PrimeIntellect, runpod, Norway EUR-NO-1)
- **vCPUs:** 15
- **RAM:** 86GB
- **CUDA:** 12.4
- **PyTorch:** 2.10.0
- **Transformers:** 5.0.0

---

## Results

### Dense Shakedown (`shake-dense`)

| Metric | Value |
|---|---|
| eval/loss | 2.4803 |
| eval/perplexity | 11.9449 |
| eval/loss_code | 1.6191 |
| eval/ppl_code | 5.0484 |
| eval/loss_math | 2.9782 |
| eval/ppl_math | 19.6527 |
| eval/loss_prose | 3.7090 |
| eval/ppl_prose | 40.8116 |
| Throughput | ~26,380 tok/s |

W&B: https://wandb.ai/sumit-ml/moe-emergence/runs/17tkzh8g

### MoE Shakedown (`shake-moe`)

| Metric | Value |
|---|---|
| eval/loss | 2.4725 |
| eval/lm_loss | 2.4577 |
| eval/lb_loss | 1.0467 |
| eval/z_loss | 4.3196 |
| eval/perplexity | 11.6784 |
| eval/loss_code | 1.5806 |
| eval/ppl_code | 4.8579 |
| eval/loss_math | 2.9725 |
| eval/ppl_math | 19.5406 |
| eval/loss_prose | 3.6998 |
| eval/ppl_prose | 40.4381 |
| Throughput | ~14,584 tok/s |

W&B: https://wandb.ai/sumit-ml/moe-emergence/runs/bl4i4el1

### Key Observations

1. MoE slightly outperforms dense even at 100 steps (eval_loss 2.47 vs 2.48),
   consistent across all three domains.
2. Load balance loss at 1.047 — near-perfect balance (ideal = 1.0), no sign of
   expert collapse.
3. Z-loss stable at ~4.3 throughout training, no instability.
4. MoE throughput is ~55% of dense (~14.6k vs ~26.4k tok/s), expected overhead from
   expert routing on a single GPU.
5. Code domain has lowest loss in both runs (~1.6), prose highest (~3.7) — consistent
   with pretrained GPT-2 strengths.

---

## Anomalies / Issues

- Token imbalance warning appeared (>1.5x ratio between domains). Expected for
  shakedown preset which uses `data-size-mb=1.0` without `--balance-tokens`.
  Budgeted runs use `--balance-tokens` per training plan.
- `loss_type=None` config warning from transformers — cosmetic, uses default
  `ForCausalLMLoss` correctly.
- No crashes, no OOM, no W&B sync issues.

---

## Cost

- **Duration:** ~5 minutes total (both runs)
- **GPU-hours:** ~0.08
- **Estimated $:** ~$0.05

---

## Conclusion

Shakedown gate **passed**. Both dense and MoE modes train correctly on the RTX 4090
with all metrics flowing to W&B. Load balancing is healthy, no expert collapse, and
throughput is reasonable. The instance and infrastructure are validated.

Ready to proceed with budgeted runs:
1. Dense baseline (~1.5hr)
2. MoE main run (~4hr)
3. No-LB ablation (~1hr, early-stop on collapse)
4. Top-2 directional (~1.5hr, optional)

---

## Artifacts

- Checkpoint (dense): `checkpoints/shake-dense/final-model.safetensors`
- Checkpoint (MoE): `checkpoints/shake-moe/final-model.safetensors`
- Logs (dense): `wandb/run-20260223_151908-17tkzh8g/`
- Logs (MoE): `wandb/run-20260223_152048-bl4i4el1/`
- W&B (dense): https://wandb.ai/sumit-ml/moe-emergence/runs/17tkzh8g
- W&B (MoE): https://wandb.ai/sumit-ml/moe-emergence/runs/bl4i4el1
