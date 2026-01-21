# Decision: Preserve Math Subdomain Labels (GSM8K vs MATH)

**Date:** 2025-01-21
**Status:** Accepted
**Context Commit:** `4b7b45e`

---

## Context

The current math loader combines GSM8K and MATH (hendrycks/competition_math) into a single "math" domain. These are conceptually distinct:

| Dataset | Content | Style |
|---------|---------|-------|
| GSM8K | Grade-school word problems | Plain English, arithmetic |
| MATH | Competition-level problems | Heavy LaTeX, symbolic logic |

If the MoE router splits these between different experts, the current pipeline merges them back into "math" for analysis, hiding potentially interesting sub-specialization. We wouldn't know if "math expert" means "arithmetic expert" vs "LaTeX expert."

This was identified as MEDIUM severity by both Opus 4.5 (Pass 1) and GPT-5.2 (Pass 1).

---

## Options Considered

### Option A: Keep Single "math" Label

**Description:** Continue merging GSM8K and MATH into one domain.

**Pros:**
- Simplest implementation
- Matches the "3 domain" framing

**Cons:**
- Cannot analyze sub-specialization
- May miss interesting routing patterns

### Option B: Add Subdomain Labels for Math Only

**Description:** Add a `subdomain` field: `"gsm8k"` or `"competition_math"`.

**Pros:**
- Enables post-hoc analysis of difficulty-based routing
- Low implementation effort
- Doesn't change the primary domain structure

**Cons:**
- Asymmetric (math has subdomains, code/prose don't)

### Option C: Add Subdomain Labels for All Domains

**Description:** Add subdomains for math AND code (language ID) AND prose (article type).

**Pros:**
- Symmetric treatment
- Maximum analysis flexibility

**Cons:**
- Code language ID requires heuristics or external tooling
- Prose subdomain is unclear (WikiText doesn't have obvious subtypes)
- Scope creep

---

## Decision

**Option B: Add Subdomain Labels for Math Only**

**Implementation:**
```python
{
    "input_ids": tensor,
    "domain": "math",
    "subdomain": "gsm8k"  # or "competition_math"
}
```

**Rationale:**
- Low effort, high analysis value
- GSM8K vs MATH is a meaningful distinction (difficulty, style)
- Code language ID is deferred — not worth the complexity unless analysis demands it
- Prose (WikiText-103) has no obvious subdomain structure

---

## Consequences

**Positive:**
- Can analyze: "Do experts split by math difficulty?"
- Can report: "Expert 3 handles 80% of GSM8K, Expert 5 handles 70% of MATH"
- No change to primary 3-domain structure

**Negative:**
- Asymmetric — code and prose don't have subdomains
- Additional field to track through the pipeline

**Future work:**
- If analysis reveals code experts splitting by language, can add language ID later
- Not blocking for initial training run

---

## References

- Multi-model debate: `docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md` (Pass 1)
- Multi-model debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md` (Pass 1)
- DATA-PIPELINE.md: GSM8K + MATH loading order documentation
