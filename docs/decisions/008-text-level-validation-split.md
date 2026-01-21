# Decision: Text-Level Validation Split Before Packing

**Date:** 2025-01-21
**Status:** Accepted
**Context Commit:** `4b7b45e`

---

## Context

The current pipeline uses 100% of data for training with no validation split. This makes it impossible to distinguish "expert specialization" from "expert memorization." Additionally, if a validation split were done at the block level (after packing), the same source document could span both train and eval blocks, causing data leakage.

The need for validation was flagged by Opus 4.5 (Pass 1) as MEDIUM-HIGH. The train/eval leakage risk from block-level splitting was caught by GPT-5.2 (Pass 8) — a critical issue that would have invalidated the validation signal entirely.

---

## Options Considered

### Option A: No Validation Split

**Description:** Use all data for training; evaluation is "separate."

**Pros:**
- Maximum training data
- Simplest implementation

**Cons:**
- Cannot prove generalization
- Specialization claims are unfalsifiable
- "Emergence" looks identical to overfitting

### Option B: Block-Level Split After Packing

**Description:** Pack all texts into blocks, then reserve some blocks for validation.

**Pros:**
- Simple to implement

**Cons:**
- **DATA LEAKAGE:** Same document can span train and eval blocks
- Invalidates the validation signal

### Option C: Text-Level Split Before Packing

**Description:** Split texts into train/eval sets first, then pack each set separately.

**Pros:**
- No document leakage
- Clean separation of train and eval data
- Validation signal is meaningful

**Cons:**
- Slightly more complex implementation
- Eval block count varies based on text lengths

### Option D: Fixed Eval Prompt Set

**Description:** Hand-pick specific examples for evaluation.

**Pros:**
- Precise control over eval content

**Cons:**
- Not scalable
- May not be representative

---

## Decision

**Option C: Text-Level Split Before Packing** with the following parameters:

**Holdout sizing formula:**
```python
n_eval_texts = min(max(10, int(len(texts) * 0.05)), int(len(texts) * 0.10))
```

This ensures:
- At least 10 texts (prevents tiny eval sets)
- Target 5% of texts
- Capped at 10% (prevents huge eval sets in small domains)

**Implementation steps:**
1. Load texts for each domain
2. Shuffle texts deterministically with seed
3. Split into train/eval at text level
4. Pack train texts → train blocks
5. Pack eval texts → eval blocks
6. Log both text counts AND block counts

**Size semantics:**
- `--size-mb` is the TOTAL (train + eval combined, pre-split)
- Eval is ~5% of total
- Train gets the remainder

---

## Consequences

**Positive:**
- No document leakage between train and eval
- Can meaningfully evaluate generalization
- Validation router entropy can be compared to training entropy
- Specialization claims become falsifiable

**Negative:**
- ~5% less training data
- Eval block counts may vary across domains (due to different text lengths)
- More complex dataloader structure

**Risks:**
- Small eval sets (especially for math) may have high variance
- Different domains may have different eval/train block ratios

**Mitigations:**
- Log text and block counts for transparency
- Use multiple seeds to verify stability of results

---

## References

- Multi-model debate: `docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md` (Pass 1, Pass 9)
- Multi-model debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md` (Pass 8 — caught leakage issue)
- Decision doc: `docs/decisions/005-phase3-data-sizing.md` (original sizing decisions)
