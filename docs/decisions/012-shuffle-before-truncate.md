# Decision: Shuffle Blocks Before Truncation for Token Balancing

**Date:** 2026-02-05
**Status:** Accepted
**Context Commit:** `5e5ebcf`

---

## Context

When `--balance-tokens` is enabled, `PackedMixedDomainDataset` truncates larger domains to match the smallest domain's block count. Currently this uses naive slicing:

```python
code_blocks = code_blocks[:min_blocks]
math_blocks = math_blocks[:min_blocks]
prose_blocks = prose_blocks[:min_blocks]
```

This systematically discards blocks from the *end* of each domain's packed sequence rather than sampling uniformly.

---

## Problem Analysis

**Current data flow:**
1. Load texts sequentially from streaming dataset (deterministic order)
2. `split_texts_for_eval()` shuffles texts, then splits train/eval
3. `pack_sequences()` packs shuffled texts into blocks (preserving order)
4. **Truncation** — `[:min_blocks]` takes first N blocks
5. Combine all domains and shuffle blocks together

**Issue:** The text-level shuffle (step 2) provides some randomization, but packing preserves that order. Truncation then discards the tail portion of blocks rather than sampling uniformly from all blocks.

**Potential bias:** If there's any systematic pattern in text lengths after shuffling (e.g., longer texts creating more blocks), truncation could introduce sampling bias.

---

## Options Considered

### Option A: Keep naive truncation (current)

**Description:** Continue using `[:min_blocks]` slicing.

**Pros:**
- Simple, deterministic
- Text-level shuffle already provides some randomization

**Cons:**
- Systematically discards tail blocks
- Could introduce bias if text lengths correlate with content

### Option B: Shuffle blocks before truncating

**Description:** Shuffle each domain's blocks before truncating.

**Pros:**
- Uniform sampling from all blocks in each domain
- Eliminates potential tail-discard bias
- Minimal code change

**Cons:**
- Must ensure proper seeding for reproducibility (see below)

---

## Decision

**Accepted: Option B** — Shuffle blocks before truncating.

The implementation is trivial and provides defensive protection against potential bias, even though empirical investigation (see below) shows the benefit is marginal for our current pipeline.

---

## CRITICAL: Reproducibility via Seeding

**This change MUST use a seeded RNG instance. Do NOT use `random.shuffle()` directly.**

Why seeding matters:
- Training runs must be reproducible for scientific validity
- Same seed + same data → same blocks selected after truncation
- Without seeding, different runs could select different block subsets

**Where to seed:** Inside `if balance_tokens` block. The shuffle is only needed when truncating — no truncation means no bias from slicing. Using a local `random.Random(seed)` instance avoids polluting global random state used by the final combined shuffle.

**Correct implementation:**

```python
if balance_tokens:
    min_blocks = min(len(code_blocks), len(math_blocks), len(prose_blocks))

    # CRITICAL: use seeded RNG for reproducibility (local instance)
    rng = random.Random(seed)
    rng.shuffle(code_blocks)
    rng.shuffle(math_blocks)
    rng.shuffle(prose_blocks)

    code_blocks = code_blocks[:min_blocks]
    math_blocks = math_blocks[:min_blocks]
    prose_blocks = prose_blocks[:min_blocks]
```

**Incorrect (DO NOT USE):**

```python
# BAD: uses global random state, not reproducible across runs
random.shuffle(code_blocks)
```

---

## Consequences

- **Positive:**
  - Uniform sampling from all blocks, not just the first N
  - Eliminates systematic tail-discard bias
  - Reproducible via seeded RNG

- **Negative:**
  - Minor: adds one shuffle operation per domain

- **Risks:**
  - If seeding is forgotten, reproducibility is broken

---

## Implementation Location

File: `moe_emergence/data.py`, inside `PackedMixedDomainDataset.__init__`

Implemented in commit following this decision's acceptance.

---

## References

- Related: `docs/decisions/005-phase3-data-sizing.md` (token balancing rationale)

---

## Empirical Investigation (2026-02-05)

Before accepting this decision, we investigated whether the shuffle-before-truncate provides meaningful benefit for our specific pipeline. The key question: **Does naive truncation introduce systematic bias?**

### Test 1: Shuffle Changes Block Selection

Verified that shuffling actually selects different blocks than naive truncation.

**Setup:** 2MB code data → 1663 blocks → truncate to 831 (half)

```
NAIVE (first N blocks):
  Block 0: [2, 15069, 2864, 383, 309, 22854, 37535, 46665, 1439, 6923]
  Block 1: [62, 6978, 62, 14894, 62, 1676, 65, 11, 2472, 62]
  Block 2: [198, 220, 220, 220, 220, 220, 220, 220, 220, 220]

SHUFFLED (random N blocks, seed=42):
  Block 0: [220, 220, 220, 220, 220, 220, 220, 220, 220, 479]
  Block 1: [7, 944, 2599, 198, 220, 220, 220, 220, 220, 220]
  Block 2: [13, 17, 22305, 198, 220, 220, 220, 220, 220, 220]

Same blocks selected: False
```

**Reproducibility confirmed:** Same seed produces identical results across runs.

### Test 2: Token Distribution — First Half vs Last Half

Checked if first-half and last-half blocks have different token distributions.

**Setup:** 3MB code data → 2479 blocks

| Metric | First Half | Last Half | Difference |
|--------|------------|-----------|------------|
| Newline tokens | 5.54% | 5.48% | 0.06% |
| Space tokens | 29.88% | 30.97% | 1.09% |
| Avg token ID | 4206.9 | 4150.1 | 56.8 (~0.1% of vocab) |

**Finding:** Minimal difference. The text-level shuffle in `split_texts_for_eval()` already randomizes ordering before packing.

### Test 3: Text Length vs Block Position

Checked if longer texts systematically create blocks at the end of the sequence.

**Setup:** 3MB code data → 783 texts → 2479 blocks

| Position | Text Count | Avg Text Length |
|----------|------------|-----------------|
| First half (blocks 0-1239) | 401 texts | 3780 chars |
| Last half (blocks 1240-2478) | 382 texts | 3894 chars |

**Difference:** 115 chars (3%) — not significant.

### Test 4: MathQA Ordering Bias

MathQA comes from a structured dataset. Checked for category/difficulty ordering.

**Setup:** 5MB math data → 13790 texts

```
First 3 texts (before shuffle):
  0: the banker's gain of a certain sum due 3 years hence...
  1: average age of students of an adult school is 40 years...
  2: sophia finished 2/3 of a book...

Last 3 texts (before shuffle):
  0: john bought 9.25 m of cloth for $425.50...
  1: a dog is tied to a tree by a long nylon cord...
  2: a can go round a circular path 8 times in 40 minutes...
```

| Position | Avg Text Length (before shuffle) | Avg Text Length (after shuffle) |
|----------|----------------------------------|--------------------------------|
| First half | 381 chars | 379 chars |
| Last half | 379 chars | 381 chars |

**Difference:** 2 chars — essentially zero ordering bias in MathQA.

### Conclusion

**The shuffle-before-truncate provides marginal benefit** for our current pipeline because:

1. `split_texts_for_eval()` already shuffles texts before packing
2. Token distributions between first/last blocks differ by <1.1%
3. Text lengths are uniformly distributed across block positions
4. MathQA has no detectable ordering bias

**Why we keep it anyway:**

1. **Defensive:** Protects against future pipeline changes (e.g., if text-level shuffle is removed)
2. **Standard practice:** Uniform sampling is statistically cleaner than tail-discard
3. **Negligible cost:** 4 lines of code, ~0ms overhead
4. **Reproducible:** Seeded RNG ensures deterministic behavior

The benefit is marginal but the cost is near-zero, making this a reasonable defensive measure.
