# Experiment: Dense Baseline

**Run ID:** run-004
**Date:** 2026-02-23
**Status:** Completed
**Type:** Training
**Phase:** 4
**Commit:** `93e5fb2`
**Command:**
```bash
uv run python -m moe_emergence.train --preset dense --run-name dense-baseline --device cuda
```

---

## Objective

Train the dense GPT-2 baseline to establish reference loss curves for comparison
against the MoE runs. This is the first of the budgeted runs from the Phase 4
training plan.

---

## Configuration

### Model
```yaml
base_model: gpt2
mode: dense
```

### Training
```yaml
preset: dense
max_steps: 5000
batch_size: 2
grad_accum_steps: 4
effective_batch: 8
block_size: 512
learning_rate: 5e-5
warmup_fraction: 0.1
eval_every: 200
save_every: 500
balance_tokens: true
```

### Data
```yaml
code_size_mb: 10
math_size_mb: 10
prose_size_mb: 10
block_size: 512
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

### Loss Progression

| Eval Step | eval/loss | perplexity | loss_code | loss_math | loss_prose |
|---|---|---|---|---|---|
| 200 | 2.5521 | 12.8341 | 1.8347 | 2.8730 | 3.5193 |
| 1400 | 2.2560 | 9.5453 | 1.6212 | 2.2441 | 3.4902 |
| 3600 | 2.1630 | 8.6975 | 1.5582 | 2.0379 | 3.4846 |
| 4999 (final) | 2.1567 | 8.6424 | 1.5538 | 2.0234 | 3.4848 |

### Final Metrics

| Metric | Value |
|---|---|
| eval/loss | **2.1567** |
| eval/perplexity | **8.6424** |
| eval/loss_code | 1.5538 |
| eval/ppl_code | 4.7295 |
| eval/loss_math | 2.0234 |
| eval/ppl_math | 7.5638 |
| eval/loss_prose | 3.4848 |
| eval/ppl_prose | 32.6165 |
| Throughput | ~25,698 tok/s |

### Key Observations

1. Clean convergence — loss decreased steadily with no spikes or instability.
2. Model largely plateaued by step ~3600, with minimal improvement in the final
   1400 steps. The cosine LR schedule was winding down by then.
3. Code domain has the lowest loss (1.55) — GPT-2 was pretrained on web text
   including code, so this is expected.
4. Prose domain is the hardest (3.48) and barely improved after step 1000.
   C4 web text is diverse and harder to memorize at this scale.
5. Math improved the most dramatically (2.87 → 2.02), suggesting the model
   can learn MathQA patterns relatively quickly.
6. Throughput stable at ~25.7k tok/s throughout.

---

## Anomalies / Issues

- None. Clean run with no crashes, OOM, or W&B sync issues.

---

## Cost

- **Duration:** ~30 minutes
- **GPU-hours:** ~0.5
- **Estimated $:** ~$0.31

---

## Conclusion

Dense baseline establishes clear reference points for MoE comparison:
- **eval/loss = 2.157** is the number to beat with MoE
- Prose domain is the hardest for both architectures (seen in shakedown too)
- Math has the most headroom for improvement

The fast convergence (~30 min vs estimated 1.5hr) is a pleasant surprise — the
RTX 4090 throughput is higher than budgeted. This leaves more budget for the
remaining runs.

Next: MoE main run (10,000 steps).

---

## Artifacts

- Checkpoint: `checkpoints/dense-baseline/final-model.safetensors`
- Resume checkpoints: `checkpoints/dense-baseline/ckpt-step-*.pt`
- Logs: `wandb/run-20260223_152549-fqhfblfv/`
- W&B: https://wandb.ai/sumit-ml/moe-emergence/runs/fqhfblfv
