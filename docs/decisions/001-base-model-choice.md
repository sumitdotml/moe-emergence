# Decision: Use GPT-2 as Base Model

**Date:** 2024-12-23 (documented retroactively)
**Status:** Accepted
**Context Commit:** `31252b6`

---

## Context

Need to choose a pretrained transformer model to convert to MoE. The goal is to demonstrate expert specialization emergence, not to achieve SOTA performance. Budget constraint: ~$80 GPU compute.

---

## Options Considered

### Option A: GPT-2 Small (124M)

**Description:** OpenAI's 2019 model. Well-documented, simple architecture.

**Pros:**
- Extremely well-understood behavior
- Simple architecture (no RoPE, no GQA, standard LayerNorm)
- HuggingFace integration is mature and stable
- Small enough to run experiments quickly
- Pretrained weights readily available

**Cons:**
- "Old" architecture (2019)
- Uses GELU, not SwiGLU
- Pre-norm but not RMSNorm

### Option B: Pythia-160M (EleutherAI)

**Description:** More modern (2023), trained on The Pile.

**Pros:**
- More recent
- Better training data
- Still small enough for budget

**Cons:**
- Less documentation on behavior
- Adds complexity without clear benefit for this study

### Option C: Custom Tiny Llama from Scratch

**Description:** Build modern architecture (RMSNorm, RoPE, SwiGLU) from scratch.

**Pros:**
- Most "modern" architecture
- Full control

**Cons:**
- High risk: budget consumed debugging training stability
- No pretrained weights - can't warm-start
- Loses the "controlled intervention" framing

---

## Decision

**GPT-2 Small** chosen for the following reasons:

1. **Controlled intervention baseline:** GPT-2's behavior is so well-documented that any observed effects can be cleanly attributed to MoE routing, not architectural quirks.

2. **Warm-start requirement:** V3 design requires experts initialized as copies of pretrained MLP. This needs pretrained weights.

3. **Budget reality:** At $80, we need experiments to work on first try. GPT-2 has the least integration risk.

4. **Framing:** "GPT-2 is old" is not a scientific criticism if properly framed as intentional baseline choice.

---

## Consequences

**Positive:**
- Simple integration with HuggingFace
- Warm-start possible via `deepcopy(block.mlp)`
- Can focus on MoE behavior rather than debugging base model

**Negative:**
- Need to defend choice in report ("why GPT-2?")
- Can't claim results are "modern"

**Mitigations:**
- V3 design includes framing guidance for report
- Optional: confirmatory run on Pythia if budget allows

---

## References

- V3 Design Doc: Part 3 (GPT-5.2's budget reality check)
- V3 Design Doc: Part 9 (Final consensus)
