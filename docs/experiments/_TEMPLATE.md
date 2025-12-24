# Experiment: [Run Name]

**Run ID:** run-NNN
**Date:** YYYY-MM-DD
**Status:** Running | Completed | Failed | Aborted
**Type:** Training | Verification | Evaluation
**Phase:** 1 | 2 | 3 | 4 | 5 | 6
**Commit:** `git-hash`
**Command:** `...`
**Script:** `path/to/script.py`

---

## Objective

What is this run trying to answer or demonstrate?

---

## Configuration

### Model
```yaml
base_model: gpt2
moe_layers: [8, 9, 10, 11]
num_experts: 8
top_k: 1
```

### Training
```yaml
epochs: 3
batch_size: 8
learning_rate: 5e-5
warmup_fraction: 0.1
max_grad_norm: 1.0
```

### Loss Coefficients
```yaml
lb_coef: 0.01    # load balancing
z_coef: 0.001    # z-loss
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

- **Hardware:** [GPU type, count]
- **CUDA:** [version]
- **PyTorch:** [version]
- **Transformers:** [version]

---

## Results

### Loss Curves

[Embed or link to wandb/tensorboard plots]

| Metric | Start | End | Best |
|--------|-------|-----|------|
| LM Loss | | | |
| LB Loss | | | |
| Z Loss | | | |

### Expert Utilization

[Per-layer utilization at end of training]

### Key Observations

1. ...
2. ...

---

## Anomalies / Issues

- [ ] Any unexpected behavior?
- [ ] Any crashes or restarts?

---

## Cost

- **Duration:** X hours
- **GPU-hours:** X
- **Estimated $:** X

---

## Conclusion

What did this run teach us? What should the next run change?

---

## Artifacts

- Checkpoint: `path/to/checkpoint`
- Logs: `path/to/logs`
- Wandb: [link]
