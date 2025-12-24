# Experiment: GPT-2 MoE Integration Verification

**Run ID:** run-001
**Date:** 2025-12-25
**Status:** Completed
**Type:** Verification
**Phase:** 2 (GPT-2 integration)
**Commit:** `1b258ffa` (dirty; modified `moe-emergence/verify_gpt2_integration.py`)
**Command:** `python moe-emergence/verify_gpt2_integration.py`
**Script:** `moe-emergence/verify_gpt2_integration.py`

---

## Objective

Verify GPT-2 integration of the MoE wrapper against the V3 spec before starting Phase 3 data work.

---

## Configuration

### Model
```yaml
base_model: gpt2 (124M)
moe_layers: [8, 9, 10, 11]
num_experts: 8
top_k: 1
noise_std: 0.1
seed: 0
```

### Training
```yaml
note: verification only (no training loop)
```

### Loss Coefficients
```yaml
note: verification only (losses computed on aux outputs)
```

### Data
```yaml
note: scripted prompts + random tokens for coverage checks
```

---

## Environment

- **Hardware:** MacBook Pro (Apple Silicon)
- **Device:** MPS
- **Runtime:** ~45 seconds
- **Python:** Not recorded for this run
- **PyTorch:** Not recorded for this run
- **Transformers:** Not recorded for this run

---

## Results

### Summary

- **Tests run:** 10
- **Tests passed:** 10
- **Failures:** 0

### Warm-Start Parity (Relative Error)

| Layer | Relative Error | Status |
| ----- | -------------- | ------ |
| 8     | 0.0027     | PASS   |
| 9     | 0.0026     | PASS   |
| 10    | 0.0022     | PASS   |
| 11    | 0.0021     | PASS   |

### Loss Checks

| Layer | Load Balance Loss | Z Loss |
| ----- | ----------------- | ------ |
| 8     | 1.1305  | 4.3161 |
| 9     | 1.1177  | 4.2913 |
| 10    | 1.1271  | 4.3142 |
| 11    | 1.0584  | 4.5191 |

### Router Gradient Norms (STE)

| Layer | Gradient Norm |
| ----- | ------------- |
| 8     | 6.048797  |
| 9     | 12.373333 |
| 10    | 16.342945 |
| 11    | 10.202866 |

### Noisy Routing Deltas

| Layer | Mean \|noisy-clean\| |
| ----- | ------------------- |
| 8     | 9.06e-03           |
| 9     | 9.80e-03           |
| 10    | 1.01e-02           |
| 11    | 1.14e-02           |

### Attention Masking

- **Max diff (batched vs single):** 9.92e-05

### Generation Sample

Prompt: "The meaning of life is"

Generated:

> The meaning of life is not the same as the meaning of death.
>
> The meaning of life is not the same as

---

## Log Excerpt

```
[Test 2] Warm-start parity...
  [OK] Layer 8 parity rel_err=0.0027
  [OK] Layer 9 parity rel_err=0.0026
  [OK] Layer 10 parity rel_err=0.0022
  [OK] Layer 11 parity rel_err=0.0021

[Test 5] Loss computation...
  [OK] Layer 8 LB=1.1305, Z=4.3161
  [OK] Layer 9 LB=1.1177, Z=4.2913
  [OK] Layer 10 LB=1.1271, Z=4.3142
  [OK] Layer 11 LB=1.0584, Z=4.5191

[Test 7] Noisy routing separation...
  [OK] Layer 8 mean |noisy-clean| = 9.06e-03
  [OK] Layer 9 mean |noisy-clean| = 9.80e-03
  [OK] Layer 10 mean |noisy-clean| = 1.01e-02
  [OK] Layer 11 mean |noisy-clean| = 1.14e-02

[Test 9] Attention masking...
  [OK] Attention masking preserved (max diff = 9.92e-05)
```

---

## Anomalies / Issues

- HuggingFace warning during loss test: `loss_type=None` unrecognized; default `ForCausalLMLoss` used.
- HuggingFace warning during generation: attention mask not inferred when pad token equals EOS.

---

## Conclusion

Phase 2 integration verification passed (10/10 tests). Proceed to Phase 3 dataset preparation once environment versions are recorded for reproducibility.

---

## Artifacts

- Script: `moe-emergence/verify_gpt2_integration.py`
- This log: `docs/experiments/run-001-gpt2-integration-verification.md`
