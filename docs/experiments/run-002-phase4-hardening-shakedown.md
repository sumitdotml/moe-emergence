# Experiment: Phase 4 Hardening Shakedown (Dense + MoE + Resume)

**Run ID:** run-002
**Date:** 2026-02-18
**Status:** Completed
**Type:** Verification
**Phase:** 4 (training infrastructure hardening)
**Commit:** `5026b2c3db2eefc06b62feb61f1dffd333aa82ec` (dirty working tree with hardening patch)
**Command:** `uv run python -m moe_emergence.train ...` (three verification runs; see below)
**Script:** `moe_emergence/train.py`

---

## Objective

Verify the pre-training hardening fixes before budgeted runs:

1. Per-domain eval metrics are computed per sequence/domain (not batch scalar copied per domain).
2. Eval perplexity is sourced from LM loss and consistent between local logs and W&B.
3. Resume path remains stable after the patch set.

---

## Configuration

### Model

```yaml
base_model: gpt2
moe_layers_dense: []
moe_layers_moe: [8, 9, 10, 11]
num_experts: 8
top_k: 1
```

### Training

```yaml
preset: shakedown
batch_size: 2
grad_accum_steps: 4
max_steps_override: 2 (dense/moe), 3 (resume check)
eval_every_override: 1
save_every_override: 1
learning_rate: 5e-5
```

### Loss Coefficients

```yaml
dense: lb_coef=0.0, z_coef=0.0
moe: lb_coef=0.01, z_coef=0.001
```

### Data

```yaml
size_mb_per_domain: 1.0
block_size: 512
balance_tokens: false (shakedown preset default)
```

---

## Environment

- **Hardware:** Apple Silicon local machine
- **Device:** MPS
- **Python:** 3.12.12
- **PyTorch:** 2.10.0
- **Transformers:** 5.0.0

---

## Results

### Commands Executed

1. `uv run python -m moe_emergence.train --preset shakedown --run-name final-dense --max-steps 2 --eval-every 1 --save-every 1 --wandb-offline`
2. `uv run python -m moe_emergence.train --preset shakedown --run-name final-moe --moe-layers 8 9 10 11 --max-steps 2 --eval-every 1 --save-every 1 --wandb-offline`
3. `uv run python -m moe_emergence.train --preset shakedown --run-name final-moe --moe-layers 8 9 10 11 --resume checkpoints/final-moe/ckpt-step-1.pt --max-steps 3 --eval-every 1 --save-every 1 --wandb-offline`

### Key Metrics (eval step)

| Run | eval/loss | eval/lm_loss | eval/perplexity | eval/loss_code | eval/loss_math | eval/loss_prose |
| --- | --- | --- | --- | --- | --- | --- |
| Dense (step 1) | 2.7172 | 2.7172 | 15.1375 | 1.7937 | 3.4939 | 3.7390 |
| MoE (step 1) | 2.7391 | 2.7242 | 15.2440 | 1.8236 | 3.4686 | 3.7365 |

### Verification Outcomes

1. **Per-domain eval fix validated:** domain losses are numerically distinct (`code != math != prose`) in both dense and MoE shakedowns.
2. **Perplexity source-of-truth validated:** `eval/perplexity` in W&B summary matches LM-based local eval value.
3. **Resume validated:** resumed run started at step 2 from `ckpt-step-1.pt`, evaluated and checkpointed successfully.

---

## Anomalies / Issues

- HuggingFace warning: tokenized sequence length warning (`1374 > 1024`) observed during data/model startup (known warning in current workflow).
- `loss_type=None` warning from transformers config fell back to default `ForCausalLMLoss` (non-blocking for this verification).

---

## Cost

- **Duration:** ~3 minutes total wall-clock for the three short verification runs
- **GPU-hours:** 0 (local MPS)
- **Estimated $:** $0 direct compute cost

---

## Conclusion

Phase 4 hardening patch is ready for budgeted training. The high-priority metric correctness issues are fixed, logging is consistent, and checkpoint resume remains stable.

---

## Artifacts

- Fix log: `docs/code-reviews/007-2026-02-18-phase4-training-fix.md`
- Review log: `docs/code-reviews/006-2026-02-18-phase4-training-review.md`
- Note: local `checkpoints/` verification artifacts were intentionally cleaned after validation.
