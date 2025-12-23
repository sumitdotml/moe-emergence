# Decision: Warm-Start Experts via deepcopy

**Date:** 2024-12-23 (documented retroactively)
**Status:** Accepted
**Context Commit:** `31252b6`

---

## Context

When replacing GPT-2's dense MLP with MoE layer, how should expert weights be initialized?

---

## Options Considered

### Option A: Random Initialization

**Description:** Initialize expert weights randomly (Xavier/Kaiming).

**Pros:**
- Simple
- Standard for training from scratch

**Cons:**
- Breaks pretrained representations
- Attention layers expect specific MLP behavior
- Model starts in broken state, must relearn basic function
- Can't distinguish "specialization" from "learning to work at all"

### Option B: deepcopy of Original MLP

**Description:** Each expert = exact copy of pretrained MLP, plus tiny noise for symmetry breaking.

**Pros:**
- Model works identically at step 0
- Experts diverge from known baseline
- Clean interpretability: "experts started identical, now differ"
- Warm-start preserves pretrained knowledge

**Cons:**
- All experts start identical (need symmetry breaking)
- Slightly more complex initialization

### Option C: Expert 0 = Original, Others Random

**Description:** Keep one expert as original, randomize others.

**Pros:**
- Guarantees one working expert

**Cons:**
- Asymmetric initialization
- Router might just always pick Expert 0
- Confounds analysis

---

## Decision

**deepcopy with tiny noise** chosen because:

1. **Interpretability:** "Experts started identical and diverged" is a cleaner narrative than "experts started random and converged/specialized."

2. **Stability:** Model produces sensible outputs from step 0. Training is stable.

3. **Warm-start validity:** The whole point of using pretrained GPT-2 is to leverage its knowledge. Random experts throw that away.

4. **V3 design validation:** Both Claude and GPT-5.2 strongly recommended this approach.

---

## Implementation Details

```python
for i in range(num_experts):
    expert = copy.deepcopy(original_mlp)

    # Symmetry breaking: tiny noise
    with torch.no_grad():
        for param in expert.parameters():
            noise = torch.randn_like(param) * param.std() * 1e-3
            param.add_(noise)
```

**Critical:** Use `param.std() * 1e-3`, NOT `param.norm() * 1e-2`. The norm is much larger (sum over millions of elements), so norm-based noise would corrupt pretrained weights.

---

## Consequences

**Positive:**
- Stable training from step 0
- Clean divergence narrative
- Preserves pretrained knowledge

**Negative:**
- Need noise for symmetry breaking (additional hyperparameter)
- All experts initially route similarly (entropy starts high)

---

## References

- V3 Design Doc: Part 4 (Warm-Start Deep Dive)
- V3 Design Doc: Part 5 (Symmetry Breaking, Noise Scale Calibration)
