# Model Debate: tracking.py Critical Review

**Date:** 2026-02-05
**File Under Review:** `moe_emergence/tracking.py`
**Participants:** CODE REVIEWER, CODE GENERATOR

---

## CODE REVIEWER - Initial Critique

### 1. Key Interface Bug: Wrong Key Name

**Issue:** `compute_router_metrics()` expects `expert_indices` but `collect_aux_outputs()` provides `topk_indices`.

```python
# tracking.py (WRONG)
indices = aux.get("expert_indices")

# gpt2_moe.py:278 (ACTUAL)
"topk_indices": moe.last_aux.topk_indices
```

**Severity:** BLOCKING — silently returns None, skipping all utilization metrics.

---

### 2. Wrong Probabilities for Entropy Calculation

**Issue:** Uses noisy `router_probs` instead of clean `router_probs_clean`.

V3 spec (lines 151-153) explicitly states:
> When using noisy routing during training, compute entropy on clean logits (no noise) for logging purposes.

**Severity:** METHODOLOGICAL ERROR — conflates exploration noise with learned routing confidence.

---

### 3. Redundant Entropy Computation

**Issue:** Router already computes entropy on clean probs (moe.py:176-178) and returns it in `aux["entropy"]`. tracking.py recomputes it incorrectly.

**Severity:** Inefficiency + incorrect result.

---

### 4. Hardcoded Layer Indices

**Issue:** Default `moe_layer_indices: list[int] = [8, 9, 10, 11]` when `aux["layer_idx"]` already exists.

**Severity:** Silent mislabeling if someone uses different MoE layers.

---

### 5. Missing Integration with Training Loop

**Issue:** Per-domain loss tracking requires knowing which samples belong to which domain — non-trivial with packed sequences. No guidance provided.

**Severity:** Documentation gap.

---

### 6. W&B Table vs Heatmap for Affinity

**Issue:** Uses `wandb.Table()` but paper needs heatmap visualization.

**Severity:** Enhancement suggestion.

---

### 7. GPU Memory Not Integrated

**Issue:** `log_gpu_memory()` exists but isn't called from `log_step()`.

**Severity:** By design (wasteful to log every step), but undocumented.

---

### 8. Positive Aspects

- Per-expert utilization logged as individual scalars (V3 spec compliant)
- Proper separation of LM/LB/Z losses
- `finish_run()` for clean W&B cleanup
- Graceful degradation when W&B unavailable

---

## CODE GENERATOR - Response

### Point 1: Wrong Key Name

**VERDICT: CODE REVIEWER is CORRECT. This is a real bug.**

Verified `gpt2_moe.py:278` — the key is `topk_indices`, not `expert_indices`.

**Action:** Fix immediately.

---

### Point 2: Noisy vs Clean Probs

**VERDICT: CODE REVIEWER is CORRECT.**

The aux dict provides both:
- `router_probs` — noisy (for training exploration)
- `router_probs_clean` — clean (for logging per V3 spec)

**Action:** Use `router_probs_clean`.

---

### Point 3: Redundant Entropy

**VERDICT: CODE REVIEWER is CORRECT.**

`aux["entropy"]` already contains pre-computed entropy on clean probs.

**Action:** Use the pre-computed value directly.

---

### Point 4: Hardcoded Layer Indices

**VERDICT: CODE REVIEWER is CORRECT.**

`aux["layer_idx"]` exists in every dict from `collect_aux_outputs()`.

**Action:** Derive dynamically from aux data.

---

### Point 5: Training Loop Integration

**VERDICT: CODE REVIEWER is CORRECT, but this is a documentation gap, not a bug.**

Per-domain loss computation belongs in Phase 4 (training infrastructure). The tracking module is a utility that accepts pre-computed values.

**Action:** Add docstring noting that caller must compute domain losses.

---

### Point 6: Table vs Heatmap

**VERDICT: Valid suggestion, but design choice.**

Tables preserve data for export. For paper figures, we'd export via W&B API and generate matplotlib heatmaps with publication styling.

**Action:** Optional enhancement — could add `wandb.plot.heatmap()` as alternative.

---

### Point 7: GPU Memory Integration

**VERDICT: This is BY DESIGN.**

GPU memory doesn't change every step — logging it constantly would be wasteful. The function exists for periodic snapshots.

**Action:** Add docstring clarifying intended usage.

---

## Consensus Summary

| Issue | Severity | Agreed? | Resolution |
|-------|----------|---------|------------|
| 1. Wrong key name | BLOCKING | ✓ Both agree | Fix: `topk_indices` |
| 2. Noisy probs | METHODOLOGICAL | ✓ Both agree | Fix: use `router_probs_clean` |
| 3. Redundant entropy | Inefficiency | ✓ Both agree | Fix: use `aux["entropy"]` |
| 4. Hardcoded layers | Silent bug | ✓ Both agree | Fix: use `aux["layer_idx"]` |
| 5. Training integration | Documentation | ✓ Both agree | Add docstring |
| 6. Table vs heatmap | Enhancement | Partial | Optional improvement |
| 7. GPU memory | Documentation | ✓ Both agree | Add docstring |

---

## Required Fixes Before Training

```python
# FIX 1: Use correct key name
indices = aux.get("topk_indices")  # was: "expert_indices"

# FIX 2: Use clean probs for any probability-based analysis
probs_clean = aux.get("router_probs_clean")  # was: "router_probs"

# FIX 3: Use pre-computed entropy
entropy = aux.get("entropy")  # was: recomputed incorrectly

# FIX 4: Derive layer index from aux data
layer_idx = aux.get("layer_idx")  # was: hardcoded [8,9,10,11]
```

---

## CODE REVIEWER - Response to Proposed Fixes

I appreciate the thorough acknowledgment, but I have concerns about **FIX 3** and the completeness of **FIX 4**.

---

### FIX 3 is Semantically Incorrect

**Problem:** The proposed fix conflates two different entropy measures.

```python
# What aux["entropy"] contains (from moe.py:176-178):
entropy = -(router_probs_clean * torch.log(router_probs_clean + 1e-9)).sum(dim=-1)
# Shape: [n_tokens] — per-token routing entropy

# What tracking.py currently computes (lines 194-195):
avg_probs = probs.mean(dim=0)  # [n_experts]
entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()
# Shape: scalar — entropy of the AVERAGED probability distribution
```

These measure **different things**:

| Metric | What it measures | Interpretation |
|--------|------------------|----------------|
| `aux["entropy"].mean()` | Average per-token uncertainty | "How confident is the router for each token?" |
| Entropy of `avg_probs` | Distribution balance across experts | "Are tokens spread evenly across experts?" |

**Example where they diverge:**
- If every token routes to exactly one expert with probability 1.0, but different tokens pick different experts uniformly...
  - Per-token entropy = 0 (each token is perfectly confident)
  - Entropy of avg_probs = log(8) ≈ 2.08 (perfect balance)

**Question for CODE GENERATOR:** Which metric does the V3 spec actually want? Line 281 says "per-layer router entropy (computed on clean logits)" — but this is ambiguous.

**My recommendation:** Log BOTH:
- `router/layer_{idx}_entropy_per_token` = `aux["entropy"].mean()`
- `router/layer_{idx}_entropy_distribution` = entropy of `avg_probs` using `router_probs_clean`

---

### FIX 4 is Incomplete

The proposed fix only changes the internal code:
```python
layer_idx = aux.get("layer_idx")
```

But the function signature still has:
```python
def compute_router_metrics(
    aux_outputs: list[dict],
    moe_layer_indices: list[int] = [8, 9, 10, 11],  # ← Still here!
) -> dict:
```

**Required changes:**
1. Remove `moe_layer_indices` parameter entirely (unused if we derive from aux)
2. Or make it truly optional with `None` default and derive when not provided
3. Update all call sites

---

### Missing: Defensive Handling

The proposed fixes use `aux.get()` which can return `None`. What happens when:
- `aux.get("topk_indices")` returns `None`?
- `aux.get("entropy")` returns `None`?

Current code would crash with `TypeError` on the next operation. Either:
1. Add explicit None checks with early return/skip
2. Or document that callers must guarantee these keys exist

---

### Updated Consensus

| Issue | Original Fix | Revised Fix |
|-------|--------------|-------------|
| 3. Entropy | Use `aux["entropy"]` | Log BOTH per-token and distribution entropy |
| 4. Layer indices | Derive from aux | Also update function signature |
| NEW | — | Add None guards or document requirements |

---

## CODE GENERATOR - Response 2

### FIX 3: Entropy Semantics

**VERDICT: CODE REVIEWER is CORRECT. I missed a crucial distinction.**

I verified by reading moe.py:176-178:
```python
entropy = -(router_probs_clean * torch.log(router_probs_clean + 1e-9)).sum(dim=-1)
# Shape: [n_tokens]
```

CODE REVIEWER's example is illuminating:
- **Per-token entropy = 0**: Router is confident for each token
- **Distribution entropy = log(8)**: Tokens spread evenly across experts

These are orthogonal measures:

| Metric | Question Answered | Use Case |
|--------|-------------------|----------|
| Per-token entropy | "Is the router confident?" | Track specialization emergence |
| Distribution entropy | "Are experts balanced?" | Monitor load balancing |

The V3 spec line 824 says "router confidence evolves" — this is **per-token entropy**.
But for load balancing analysis, we need **distribution entropy**.

**Accepted:** Log BOTH metrics as CODE REVIEWER suggests.

---

### FIX 4: Function Signature

**VERDICT: CODE REVIEWER is CORRECT. My fix was incomplete.**

If we derive `layer_idx` from aux data, the `moe_layer_indices` parameter becomes dead code.

**Accepted:** Remove the parameter entirely. The aux dict is the source of truth.

---

### Defensive Handling

**VERDICT: CODE REVIEWER is CORRECT.**

`aux.get()` returning `None` would cause crashes. Two options:

1. **Explicit guards** (more defensive)
2. **Document requirements** (cleaner API)

Given that `collect_aux_outputs()` in gpt2_moe.py always populates these keys when `last_aux` exists, I lean toward **Option 2** + a single top-level guard:

```python
def compute_router_metrics(aux_outputs: list[dict]) -> dict:
    """
    ...
    Note: aux dicts must contain keys from collect_aux_outputs():
        layer_idx, router_probs_clean, topk_indices, entropy
    """
    metrics = {}
    if not aux_outputs:
        return metrics

    for aux in aux_outputs:
        if not all(k in aux for k in ["layer_idx", "topk_indices", "entropy"]):
            continue  # skip malformed entries
        ...
```

This handles malformed input gracefully without excessive per-field guards.

---

## Final Consensus

| Issue | Final Resolution |
|-------|-----------------|
| 1. Key name | `topk_indices` (agreed) |
| 2. Clean probs | `router_probs_clean` (agreed) |
| 3. Entropy | Log BOTH: `entropy_per_token` and `entropy_distribution` |
| 4. Layer indices | Remove parameter, derive from `aux["layer_idx"]` |
| 5. Defensive handling | Document requirements + skip malformed entries |

---

## CODE REVIEWER - Final Review

I accept the CODE GENERATOR's Response 2. The fixes are now complete and well-reasoned.

### Points of Agreement

**FIX 3 (Entropy):** Logging both metrics is the correct solution. The distinction between per-token confidence and distribution balance is important for interpreting MoE behavior.

**FIX 4 (Signature):** Removing the parameter entirely is cleaner than making it optional. The aux dict is indeed the source of truth.

**Defensive Handling:** The proposed approach is pragmatic:
```python
if not all(k in aux for k in ["layer_idx", "topk_indices", "entropy"]):
    continue  # skip malformed entries
```
This is better than excessive per-field guards, and documenting the contract in the docstring is sufficient.

---

### One Minor Clarification

The defensive check should also include `router_probs_clean` since we need it for distribution entropy:

```python
required_keys = ["layer_idx", "topk_indices", "entropy", "router_probs_clean"]
if not all(k in aux for k in required_keys):
    continue
```

Otherwise we'd still crash when computing `avg_probs = probs_clean.mean(dim=0)` on `None`.

---

### Verdict

**CONSENSUS REACHED.** No further objections.

The fixes are well-designed and address all identified issues. Ready for implementation.

---

## Status

**Consensus reached.** Both parties agree on final fixes.

**Summary of Required Changes:**

| # | Change | Location |
|---|--------|----------|
| 1 | `expert_indices` → `topk_indices` | `compute_router_metrics()` |
| 2 | `router_probs` → `router_probs_clean` | `compute_router_metrics()` |
| 3 | Log both `entropy_per_token` and `entropy_distribution` | `compute_router_metrics()` |
| 4 | Remove `moe_layer_indices` parameter | `compute_router_metrics()` signature |
| 5 | Add required keys guard + docstring | `compute_router_metrics()` |

**Next step:** Apply fixes to `moe_emergence/tracking.py`

---

## CODE GENERATOR - Final Acknowledgment

**CODE REVIEWER's clarification is correct.** The required keys must include `router_probs_clean` for distribution entropy computation.

**Accepted.** No objections.

---

## CONVERGENCE SUMMARY

**Debate concluded:** 2026-02-05

### Final Agreed Fixes

| # | Change | Details |
|---|--------|---------|
| 1 | Key name | `expert_indices` → `topk_indices` |
| 2 | Clean probs | `router_probs` → `router_probs_clean` |
| 3 | Dual entropy | Log `entropy_per_token` (router confidence) AND `entropy_distribution` (load balance) |
| 4 | Dynamic layers | Remove `moe_layer_indices` param, use `aux["layer_idx"]` |
| 5 | Defensive guard | Check for `["layer_idx", "topk_indices", "entropy", "router_probs_clean"]` |

### Key Insight from Debate

The entropy distinction (Point 3) was non-obvious but important:
- **Per-token entropy** answers: "Is the router confident?"
- **Distribution entropy** answers: "Are tokens spread evenly?"

Logging both enables proper analysis of specialization vs load balancing.

### Status

**IMPLEMENTED** — All fixes applied to `moe_emergence/tracking.py`

### Post-Implementation Note

CODE REVIEWER raised a minor observation about normalizing `entropy_per_token` for consistency. CODE GENERATOR's response:

> Distribution entropy benefits from normalization (comparing to ideal balance of 1.0). Per-token entropy is tracked as a trend over training — raw values suffice.

**Accepted.** No changes needed.
