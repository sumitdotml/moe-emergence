# Refactor: Deduplicate Loss Helpers & Harden Tests

**Date:** 2025-12-23
**Commit:** `c929d8c`
**Reviewer:** GPT-5.2
**Previous Commit:** `ec90acb`

---

## Summary

GPT-5.2 identified code drift risk from duplicate loss function implementations and brittle tests. Changes consolidate loss helpers in `moe.py` and strengthen the `gpt2_moe.py` smoke test.

---

## Changes Applied

### 1. Deduplicated Loss Helpers

**Problem:**
`compute_load_balance_loss` and `compute_z_loss` were defined in both `moe.py` (as class methods) and `gpt2_moe.py` (as standalone functions). This creates drift risk—fixes applied to one location may not propagate to the other.

**Solution:**

- Moved both functions to module level in `moe.py`
- `gpt2_moe.py` now imports them: `from moe import Router, compute_load_balance_loss, compute_z_loss`
- `MoE` class calls module-level functions instead of methods

**Files Changed:**

- `moe.py`: `compute_load_balance_loss` and `compute_z_loss` moved outside `MoE` class (lines 299-428)
- `gpt2_moe.py`: Removed duplicate implementations, added import (line 20)

---

### 2. Hardened Dummy MLP Test

**Problem:**
Original test had:

- Hard-coded shape assertions (brittle)
- No checks for `router_probs_clean`, `router_logits`, `entropy`
- No gradient flow verification for STE

**Solution:**
Added comprehensive assertions:

```python
# Shape assertions using computed values
n_tokens = x.shape[0] * x.shape[1]
n_experts = moe.n_experts
topk = moe.topk

assert aux.router_probs.shape == (n_tokens, n_experts)
assert aux.router_probs_clean.shape == (n_tokens, n_experts)
assert aux.router_logits.shape == (n_tokens, n_experts)
assert aux.topk_indices.shape == (n_tokens, topk)
assert aux.topk_weights.shape == (n_tokens, topk)
assert aux.entropy.shape == (n_tokens,)

# Index bounds check
assert aux.topk_indices.min() >= 0
assert aux.topk_indices.max() < n_experts

# Weights sum to 1
assert torch.allclose(aux.topk_weights.sum(dim=-1), torch.ones(n_tokens), atol=1e-6)

# Loss sanity checks
assert lb_loss.ndim == 0, "Load balance loss is not scalar"
assert z_loss.ndim == 0, "Z-loss is not scalar"
assert torch.isfinite(lb_loss), "Load balance loss is not finite"
assert torch.isfinite(z_loss), "Z-loss is not finite"

# STE gradient check
loss = output.mean()
loss.backward()
router_grad = moe.router.gate.weight.grad
assert router_grad is not None, "Router has no gradient"
assert router_grad.abs().sum() > 0, "Router gradient is zero"
```

---

### 3. Comment Cleanup

Standardized comment style using `:::` markers for section headers:

- `:::warm-start:::`
- `:::symmetry breaking:::`
- `:::storage for auxiliary outputs:::`

---

## Remaining Gaps (Optional/Scale-Related)

Per GPT-5.2's assessment, the following are **not required for V3 spec** but may be useful at scale:

| Gap                     | Description                                    | Priority                       |
| ----------------------- | ---------------------------------------------- | ------------------------------ |
| 2D input support        | Router assumes 3D input `[batch, seq, hidden]` | Optional                       |
| top-k > n_experts guard | No validation if `topk > n_experts`            | Optional                       |
| Faster expert batching  | Loop-based dispatch is O(n_experts)            | Optional (fine at small scale) |

---

## Verification

- [x] `python moe-emergence/gpt2_moe.py` passes (smoke test)
- [x] `python moe-emergence/training_demo.py` passes
- [x] No regressions in loss values or gradient flow

---

## References

- Previous review: `003-2025-12-23-gpt2-moe-fix.md`
- V3 Design Doc: `project-design/MOE-PROJECT-DESIGN-V3.md`
