# Decision: Randomize Math Problem/Solution Prefixes

**Date:** 2025-01-21
**Status:** Accepted
**Context Commit:** `4b7b45e`

---

## Context

The current math data loader formats every example as:
```
Problem: {question}

Solution: {answer}
```

This creates a routing shortcut: MoE routers are "lazy" and look for the strongest signal. The token `Problem:` at the start of every math sequence is a stronger signal than abstract mathematical reasoning. An expert might achieve 99% routing accuracy on math simply by activating on `Problem:` — "cheat emergence" rather than semantic specialization.

This was identified as a HIGH severity issue by GPT-5.2 (Pass 1) and independently confirmed by Gemini 3.

---

## Options Considered

### Option A: Keep Fixed Prefixes

**Description:** Keep `Problem:` / `Solution:` for all examples.

**Pros:**
- Simpler implementation
- Clear structure for human inspection

**Cons:**
- Router learns prefix tokens, not math content
- Invalidates "math specialization" claims

### Option B: Remove All Prefixes

**Description:** Concatenate question + answer with only a separator.

**Pros:**
- Eliminates routing shortcuts entirely

**Cons:**
- Loses problem/solution boundary for downstream analysis
- Cannot analyze "does router behave differently on problem vs solution tokens?"

### Option C: Randomize Both Prefixes

**Description:** Randomly select from a small set of non-empty prefixes for both problem and solution.

**Pros:**
- Breaks single-token routing shortcut
- Preserves boundary markers for analysis
- Forces router to look at content, not prefix

**Cons:**
- Slightly more complex implementation
- Introduces variation in data format

### Option D: Keep Prefixes, Exclude in Analysis

**Description:** Keep fixed prefixes but exclude first N tokens when analyzing routing.

**Pros:**
- No data format change

**Cons:**
- Post-hoc workaround — doesn't prevent router from learning shortcuts
- Complicates analysis

---

## Decision

**Option C: Randomize Both Prefixes** with the following sets:

```python
PROBLEM_PREFIXES = ["Problem:", "Question:", "Given:"]
SOLUTION_PREFIXES = ["Solution:", "Answer:", "Therefore:"]
```

**Important constraints:**
- **No empty strings** — empty prefixes would remove boundary cues entirely
- **Do not log per-example prefix choices** — only log the prefix sets used (avoid scope creep)
- Use seeded random selection for reproducibility

**Implementation:**
```python
import random
rng = random.Random(seed)
problem_prefix = rng.choice(PROBLEM_PREFIXES)
solution_prefix = rng.choice(SOLUTION_PREFIXES)
text = f"{problem_prefix} {question}\n\n{solution_prefix} {answer}"
```

---

## Consequences

**Positive:**
- Router cannot rely on a single invariant token for math routing
- Boundary markers preserved for potential problem-vs-solution analysis
- Reproducible with seed

**Negative:**
- Data format varies across examples (minor)
- Analysis must account for prefix variation if examining exact token routing

**Risks:**
- Router might still learn that any of these prefixes → math, but this is a weaker signal than a single invariant token

---

## References

- Multi-model debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md` (Pass 1, Pass 4-5)
- Multi-model debate: `docs/models-debate/005c-DATA-PIPELINE-CRITICAL-REVIEW-GEMINI-3.md`
- GPT-5.2 Pass 5 caught the "Solution:" oversight — both markers must be randomized
