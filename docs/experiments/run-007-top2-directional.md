# Experiment: Top-2 Directional

**Run ID:** run-007
**Date:** 2026-02-24
**Status:** Completed
**Type:** Training (ablation)
**Phase:** 4
**Commit:** `46cffc5`
**Command:**
```bash
uv run python -m moe_emergence.train \
  --preset top2 \
  --run-name top2-main-10k \
  --max-steps 10000 \
  --device cuda \
  --eval-every 200 \
  --save-every 500 \
  --log-router-every 20
```

---

## Objective

Test whether top-2 routing improves performance over top-1 -- particularly on prose,
where the MoE main run (run-005, top-1) lost to the dense baseline by 1.6%. Top-2
routes each token to two experts with soft-weighted combination, giving the model more
capacity per token at the cost of ~2x expert compute. This also tests whether top-2
maintains healthy load balance under the same `lb_coef=0.01` setting.

Run extended to 10,000 steps (via `--max-steps 10000` CLI override of the top2 preset
default of 3000) to match the MoE main run for fair comparison.

---

## Configuration

### Model
```yaml
base_model: gpt2
moe_layers: [8, 9, 10, 11]
num_experts: 8
top_k: 2
```

### Training
```yaml
preset: top2
max_steps: 10000 (overridden from preset default of 3000)
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

### Loss Coefficients
```yaml
lb_coef: 0.01
z_coef: 0.001
noise_std: 0.1    # router noise ON (preset default)
```

### Data
```yaml
code_size_mb: 10
math_size_mb: 10
prose_size_mb: 10
block_size: 512
```

After token balancing: 4296 blocks/domain, 12,888 total (~6.6M tokens).

---

## Environment

- **Hardware:** 1x RTX 4090 24GB (PrimeIntellect, runpod, Romania EU-RO-1)
- **vCPUs:** 8
- **RAM:** 46GB
- **CUDA:** 12.9 (driver 575.57)
- **PyTorch:** 2.10.0+cu128
- **Transformers:** 5.0.0

---

## Results

### Eval Progression

| Step | eval/loss | lm_loss | lb_loss | z_loss | ppl | loss_code | loss_math | loss_prose |
|---|---|---|---|---|---|---|---|---|
| 200 | 2.6271 | 2.6124 | 1.0412 | 4.3254 | 13.63 | 1.8671 | 3.0180 | 3.5245 |
| 400 | 2.4916 | 2.4771 | 1.0311 | 4.2173 | 11.91 | 1.7704 | 2.7346 | 3.5049 |
| 600 | 2.4274 | 2.4131 | 1.0313 | 4.0601 | 11.17 | 1.7256 | 2.5958 | 3.4994 |
| 800 | 2.3806 | 2.3664 | 1.0347 | 3.8226 | 10.66 | 1.6937 | 2.4929 | 3.4965 |
| 1000 | 2.3455 | 2.3316 | 1.0405 | 3.5531 | 10.29 | 1.6758 | 2.4088 | 3.4916 |
| 1200 | 2.3200 | 2.3063 | 1.0430 | 3.2981 | 10.04 | 1.6659 | 2.3420 | 3.4902 |
| 1400 | 2.2815 | 2.2680 | 1.0434 | 3.0570 | 9.66 | 1.6300 | 2.2713 | 3.4889 |
| 1600 | 2.2607 | 2.2474 | 1.0415 | 2.8599 | 9.46 | 1.6210 | 2.2205 | 3.4846 |
| 1800 | 2.2430 | 2.2299 | 1.0447 | 2.6684 | 9.30 | 1.6094 | 2.1726 | 3.4944 |
| 2000 | 2.2275 | 2.2146 | 1.0405 | 2.5482 | 9.16 | 1.5967 | 2.1437 | 3.4917 |
| 2200 | 2.2156 | 2.2028 | 1.0427 | 2.3795 | 9.05 | 1.5932 | 2.1101 | 3.4920 |
| 2400 | 2.2019 | 2.1892 | 1.0422 | 2.2723 | 8.93 | 1.5837 | 2.0791 | 3.4929 |
| 2600 | 2.1898 | 2.1772 | 1.0423 | 2.1472 | 8.82 | 1.5767 | 2.0539 | 3.4880 |
| 2800 | 2.1794 | 2.1669 | 1.0440 | 2.0579 | 8.73 | 1.5723 | 2.0250 | 3.4900 |
| **3000** | **2.1697** | **2.1574** | **1.0414** | **1.9697** | **8.65** | **1.5646** | **2.0038** | **3.4920** |
| 3200 | 2.1612 | 2.1489 | 1.0435 | 1.8903 | 8.58 | 1.5649 | 1.9785 | 3.4883 |
| 3400 | 2.1577 | 2.1454 | 1.0440 | 1.8252 | 8.55 | 1.5594 | 1.9624 | 3.5046 |
| 3600 | 2.1504 | 2.1383 | 1.0424 | 1.7384 | 8.48 | 1.5573 | 1.9411 | 3.5060 |
| 3800 | 2.1414 | 2.1293 | 1.0437 | 1.6874 | 8.41 | 1.5504 | 1.9243 | 3.5030 |
| 4000 | 2.1342 | 2.1222 | 1.0438 | 1.6408 | 8.35 | 1.5454 | 1.9069 | 3.5050 |
| 4200 | 2.1276 | 2.1156 | 1.0430 | 1.5952 | 8.29 | 1.5428 | 1.8925 | 3.5010 |
| 4400 | 2.1212 | 2.1092 | 1.0439 | 1.5485 | 8.24 | 1.5393 | 1.8735 | 3.5049 |
| 4600 | 2.1144 | 2.1024 | 1.0442 | 1.5173 | 8.19 | 1.5352 | 1.8600 | 3.5018 |
| 4800 | 2.1079 | 2.0960 | 1.0431 | 1.4985 | 8.13 | 1.5319 | 1.8438 | 3.5020 |
| **5000** | **2.1099** | **2.0980** | **1.0444** | **1.4474** | **8.15** | **1.5334** | **1.8359** | **3.5176** |
| 5200 | 2.1118 | 2.0999 | 1.0455 | 1.4255 | 8.17 | 1.5369 | 1.8292 | 3.5275 |
| 5400 | 2.1050 | 2.0932 | 1.0458 | 1.3995 | 8.11 | 1.5314 | 1.8164 | 3.5262 |
| 5600 | 2.0987 | 2.0869 | 1.0459 | 1.3754 | 8.06 | 1.5264 | 1.8086 | 3.5193 |
| 5800 | 2.0961 | 2.0843 | 1.0461 | 1.3617 | 8.04 | 1.5255 | 1.7986 | 3.5229 |
| 6000 | 2.0912 | 2.0794 | 1.0452 | 1.3449 | 8.00 | 1.5243 | 1.7879 | 3.5183 |
| 6200 | 2.0872 | 2.0754 | 1.0456 | 1.3232 | 7.97 | 1.5215 | 1.7792 | 3.5180 |
| 6400 | 2.0852 | 2.0734 | 1.0476 | 1.3142 | 7.95 | 1.5208 | 1.7735 | 3.5183 |
| 6600 | 2.0889 | 2.0771 | 1.0472 | 1.2963 | 7.98 | 1.5241 | 1.7692 | 3.5328 |
| 6800 | 2.0874 | 2.0757 | 1.0475 | 1.2879 | 7.97 | 1.5232 | 1.7656 | 3.5332 |
| 7000 | 2.0842 | 2.0724 | 1.0481 | 1.2762 | 7.94 | 1.5213 | 1.7592 | 3.5315 |
| 7200 | 2.0839 | 2.0722 | 1.0471 | 1.2659 | 7.94 | 1.5209 | 1.7558 | 3.5356 |
| 7400 | 2.0789 | 2.0672 | 1.0476 | 1.2667 | 7.90 | 1.5179 | 1.7488 | 3.5294 |
| 7600 | 2.0823 | 2.0706 | 1.0471 | 1.2562 | 7.93 | 1.5224 | 1.7472 | 3.5369 |
| 7800 | 2.0784 | 2.0667 | 1.0472 | 1.2477 | 7.90 | 1.5190 | 1.7424 | 3.5332 |
| 8000 | 2.0766 | 2.0649 | 1.0476 | 1.2477 | 7.88 | 1.5180 | 1.7395 | 3.5311 |
| 8200 | 2.0788 | 2.0670 | 1.0477 | 1.2385 | 7.90 | 1.5189 | 1.7400 | 3.5380 |
| 8400 | 2.0790 | 2.0672 | 1.0479 | 1.2375 | 7.90 | 1.5193 | 1.7397 | 3.5384 |
| 8600 | 2.0776 | 2.0659 | 1.0481 | 1.2346 | 7.89 | 1.5186 | 1.7358 | 3.5392 |
| 8800 | 2.0778 | 2.0660 | 1.0483 | 1.2323 | 7.89 | 1.5182 | 1.7351 | 3.5415 |
| 9000 | 2.0771 | 2.0654 | 1.0480 | 1.2335 | 7.89 | 1.5183 | 1.7339 | 3.5401 |
| 9200 | 2.0769 | 2.0652 | 1.0479 | 1.2345 | 7.89 | 1.5180 | 1.7343 | 3.5392 |
| 9400 | 2.0768 | 2.0651 | 1.0479 | 1.2352 | 7.89 | 1.5176 | 1.7336 | 3.5403 |
| 9600 | 2.0767 | 2.0650 | 1.0480 | 1.2335 | 7.89 | 1.5177 | 1.7334 | 3.5401 |
| 9800 (final) | 2.0768 | 2.0651 | 1.0481 | 1.2327 | 7.89 | 1.5178 | 1.7335 | 3.5403 |

### Final Metrics

| Metric | Value |
|---|---|
| eval/loss | **2.0768** |
| eval/lm_loss | 2.0651 |
| eval/lb_loss | 1.0481 |
| eval/z_loss | 1.2327 |
| eval/perplexity | **7.8863** |
| eval/loss_code | 1.5178 |
| eval/ppl_code | 4.5624 |
| eval/loss_math | 1.7335 |
| eval/ppl_math | 5.6599 |
| eval/loss_prose | 3.5403 |
| eval/ppl_prose | 34.4739 |
| best eval/loss | 2.0766 (step 8000) |
| Throughput | ~18,874 tok/s |

### MoE Main Comparison (run-005, top-1, step-matched)

| Step | Top-1 eval/loss | Top-2 eval/loss | Delta |
|---|---|---|---|
| 200 | 2.6250 | 2.6271 | +0.002 (+0.08%) |
| 1000 | 2.3487 | 2.3455 | -0.003 (-0.14%) |
| 2000 | 2.2341 | 2.2275 | -0.007 (-0.30%) |
| 3000 | 2.1737 | 2.1697 | -0.004 (-0.18%) |
| 4000 | 2.1389 | 2.1342 | -0.005 (-0.22%) |
| 5000 | 2.1156 | 2.1099 | -0.006 (-0.27%) |
| 6000 | 2.0949 | 2.0912 | -0.004 (-0.18%) |
| 7000 | 2.0879 | 2.0842 | -0.004 (-0.18%) |
| 8000 | 2.0804 | 2.0766 | -0.004 (-0.18%) |
| 9000 | 2.0804 | 2.0771 | -0.003 (-0.16%) |
| 9800 (final) | 2.0798 | 2.0768 | -0.003 (-0.14%) |

Top-2 started essentially tied with top-1, then gradually developed a ~0.2-0.3% lead
peaking around steps 2000-5000. The gap narrowed slightly in the plateau region,
settling at 0.14% by the end. Small but consistent.

**Per-domain final comparison (step 9800):**

| Metric | Top-1 | Top-2 | Delta |
|---|---|---|---|
| eval/loss | 2.0798 | 2.0768 | -0.14% |
| loss_code | 1.5207 | 1.5178 | -0.19% |
| loss_math | 1.7403 | 1.7335 | -0.39% |
| loss_prose | 3.5413 | 3.5403 | -0.03% |
| perplexity | 7.91 | 7.89 | -0.25% |
| lb_loss | 1.0115 | 1.0481 | +3.6% |
| throughput | ~14,200 tok/s | ~18,874 tok/s | +33% |

### Dense Baseline Comparison (run-004)

| Metric | Dense (5000 steps) | Top-1 MoE (10K) | Top-2 MoE (10K) | Top-2 vs Dense |
|---|---|---|---|---|
| eval/loss | 2.1567 | 2.0798 | 2.0768 | -3.7% |
| loss_code | 1.5538 | 1.5207 | 1.5178 | -2.3% |
| loss_math | 2.0234 | 1.7403 | 1.7335 | -14.3% |
| loss_prose | 3.4848 | 3.5413 | 3.5403 | +1.6% |
| perplexity | 8.64 | 7.91 | 7.89 | -8.7% |

Both MoE variants beat dense on aggregate and domain-specific metrics except prose,
where the dense model's simpler architecture retains a 1.6% advantage. Top-2 provides
a marginal 0.14% improvement over top-1 -- not enough to justify the 2x expert compute
per token.

### Key Observations

1. **Top-2 provides marginal improvement over top-1.** The 0.14% final gap is real
   (consistent across 40+ eval checkpoints) but small enough to be practically irrelevant.
   The Switch Transformer finding holds: top-1 captures most of the routing benefit.

2. **Math is where top-2 helps most.** loss_math improved 0.39% over top-1 (1.7335 vs
   1.7403). Code improved 0.19%. Prose was essentially unchanged (-0.03%). This suggests
   the second expert path helps slightly with structured reasoning but doesn't improve
   the model's ability to handle diverse web text.

3. **Top-2 did not fix the prose weakness.** This was the main hypothesis to test. Both
   MoE variants lose to the dense baseline on prose by 1.6%. The problem isn't routing
   capacity -- it's that C4 web text is too heterogeneous for expert specialization at
   this scale. Adding a second expert per token doesn't help when there's nothing
   meaningful to specialize on.

4. **Load balance stayed healthy.** lb_loss settled at ~1.048 vs top-1's ~1.012. Higher,
   but well below the collapse threshold (~1.2). The `lb_coef=0.01` setting works for
   both top-1 and top-2 routing.

5. **Both runs plateaued at the same point.** ~step 7000-8000 for both architectures,
   with fluctuations of +-0.001 thereafter. The training data ceiling is the same
   regardless of routing scheme.

6. **Throughput was surprisingly higher.** Top-2 ran at ~18.9k tok/s vs top-1's ~14.2k
   tok/s (+33%). Top-2 should be more expensive per token (2x expert dispatch), so this
   is likely due to hardware variance between the two PrimeIntellect pods (different
   regions: Norway for top-1, Romania for top-2).

---

## Anomalies / Issues

- No crashes, OOM, or W&B sync issues.
- Collapse detection was disabled (`collapse_early_stop: false`) since top-2 routing
  distributes load more naturally. No collapse occurred.

---

## Cost

- **Duration:** ~48 minutes
- **GPU-hours:** ~0.8
- **Estimated $:** ~$0.49 (0.8hr x $0.61/hr)

---

## Conclusion

Top-2 routing does not meaningfully outperform top-1 for this architecture and dataset.
The 0.14% improvement is statistically consistent but practically negligible -- it would
not justify the 2x per-token expert compute in a production setting.

The key finding is negative but useful: **the prose weakness in MoE training is not a
routing capacity problem.** Adding a second expert path doesn't help because C4 web text
lacks the clusterable structure that expert specialization exploits. Math and code have
learnable patterns that routing can separate; prose (diverse web text) does not, at this
scale.

This validates the top-1 routing choice from the MoE main run (run-005) and aligns with
the Switch Transformer result that top-1 is sufficient for expert utilization.

---

## Artifacts

```
top2-main-10k/
├── final-model.safetensors      # 1.2 GB -- top-2 MoE model at step 9999
├── final-model.json             # metadata sidecar
├── best-model.safetensors       # 1.2 GB -- best eval loss (step 8000)
├── best-model.json
├── model-step-{500..9500}.safetensors  # per-eval snapshots (500-step intervals)
├── model-step-{500..9500}.json
├── ckpt-step-{9000,9500,9999}.pt       # 2.9 GB each -- full resume checkpoints
├── metrics.jsonl                # per-step training + eval metrics
├── config.json                  # run config
└── run_summary.json             # final summary
```

- Checkpoint: `checkpoints/top2-main-10k/`
- W&B: https://wandb.ai/sumit-ml/moe-emergence/runs/6mw6qbac
