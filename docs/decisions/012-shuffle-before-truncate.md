# Decision: Shuffle Blocks Before Truncation for Token Balancing

**Date:** 2026-02-05
**Status:** Proposed
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

**Proposed: Option B** — Shuffle blocks before truncating.

The implementation is trivial and eliminates a potential source of bias.

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

See TODO comment in code referencing this decision.

---

## References

- Related: `docs/decisions/005-phase3-data-sizing.md` (token balancing rationale)
- TODO in code: `moe_emergence/data.py:421-423`
