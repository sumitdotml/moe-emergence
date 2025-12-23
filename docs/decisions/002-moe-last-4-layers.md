# Decision: MoE in Last 4 Layers Only

**Date:** 2024-12-23 (documented retroactively)
**Status:** Accepted
**Context Commit:** `31252b6`

---

## Context

GPT-2 has 12 transformer layers. Need to decide which layers to convert to MoE.

---

## Options Considered

### Option A: All 12 Layers

**Pros:**
- Maximum capacity increase
- More data points for analysis

**Cons:**
- 3x compute cost vs last 4 layers
- Early layers may not benefit (learn generic features)
- Consumes budget, leaves less for ablations

### Option B: Last 4 Layers (8, 9, 10, 11)

**Pros:**
- Later layers learn domain-specific features (more likely to specialize)
- 2/3 compute savings vs all layers
- Leaves budget for required ablations

**Cons:**
- Fewer MoE layers to analyze
- Might miss interesting early-layer patterns

### Option C: Alternating Layers

**Pros:**
- Mix of early and late layer data

**Cons:**
- Harder to interpret
- No clear scientific rationale

---

## Decision

**Last 4 layers (8-11)** chosen because:

1. **Scientific rationale:** Research suggests early transformer layers learn generic features (syntax, common patterns) while later layers learn semantic/domain-specific features. Specialization is most meaningful where features are already differentiated.

2. **Budget efficiency:** Reduces training compute by ~2/3, leaving room for required ablations (dense baseline, no-LB collapse, top-2 directional).

3. **Cleaner analysis:** Specialization in later layers is more interpretable ("this expert handles code semantics") vs early layers ("this expert handles... subword patterns?").

---

## Consequences

**Positive:**
- Faster training iterations
- Budget for ablations
- Cleaner specialization narrative

**Negative:**
- Can't claim insights about early-layer MoE behavior
- Total parameter count lower than all-layer MoE

**Mitigations:**
- Acknowledge limitation in report
- Frame as "focused study on semantic-level specialization"

---

## References

- V3 Design Doc: Part 4 (MoE Only in Last N Layers)
- Various papers showing layer-wise feature hierarchy in transformers
