# Decision: Streaming Shuffle Buffer for CodeParrot

**Date:** 2025-01-21
**Status:** Accepted
**Context Commit:** `4b7b45e`

---

## Context

CodeParrot is loaded via streaming (`streaming=True`) and the current implementation takes the first N MB of data by breaking after reaching the target size. This creates a sampling bias: streaming datasets iterate in a fixed order (by repo ID, file size, or index position), so the first 10MB is not representative of the full dataset.

This was identified as a HIGH severity issue in the multi-model debate (Opus 4.5 + GPT-5.2 + Gemini 3, 11 passes). Gemini 3 called it the "Alphabetical Trap" — the code domain might actually be "Django Expert" or "Unit Test Expert" depending on what repos appear first.

---

## Options Considered

### Option A: Accept Bias and Document

**Description:** Keep current behavior, document the limitation.

**Pros:**
- No code changes needed
- Simplest approach

**Cons:**
- Invalidates "code specialization" claims
- Results are artifacts of dataset ordering, not true domain patterns

### Option B: Load 2x, Shuffle in Memory, Truncate

**Description:** Load more data than needed, shuffle in memory, then truncate.

**Pros:**
- Simple to implement

**Cons:**
- Still samples from the START of the stream — doesn't fix the bias
- Memory overhead

### Option C: Streaming Shuffle Buffer

**Description:** Use `ds.shuffle(buffer_size=N, seed=S)` to shuffle while streaming.

**Pros:**
- Proper randomization without loading entire dataset
- Memory-efficient (only buffer_size examples in memory)
- Reproducible with seed
- Supported by HuggingFace `datasets` library

**Cons:**
- Buffer size is a heuristic, not a guarantee of unbiased sampling
- Larger buffers use more memory

### Option D: Reservoir Sampling

**Description:** Implement true reservoir sampling for uniform random selection.

**Pros:**
- Theoretically optimal for random sampling from stream

**Cons:**
- More complex to implement
- Overkill for this project's scale

---

## Decision

**Option C: Streaming Shuffle Buffer** with the following parameters:

```python
buffer_size = max(1000, int(size_mb * 200))
ds = ds.shuffle(buffer_size=buffer_size, seed=args.seed)
```

**Rationale:**
- Balances randomization quality with simplicity
- Buffer scales with target size (at 10MB: buffer=2000)
- Seed ensures reproducibility
- Formula assumes ~200 examples per MB (rough heuristic for filtered CodeParrot)

---

## Consequences

**Positive:**
- Significantly reduces ordering bias
- Reproducible with seed
- Minimal memory overhead

**Negative:**
- Buffer size is a heuristic — small buffers can still be biased
- Adds a CLI parameter to track/log

**Risks:**
- If upstream dataset ordering changes, results may differ even with same seed
- Buffer size formula assumes ~5KB per example; actual ratio may vary

**Mitigations:**
- Log buffer_size in every run for reproducibility
- Document that this is bias-reduction, not bias-elimination

---

## References

- Multi-model debate: `docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md` (Pass 3-11)
- Multi-model debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md` (Pass 1)
- Multi-model debate: `docs/models-debate/005c-DATA-PIPELINE-CRITICAL-REVIEW-GEMINI-3.md`
- HuggingFace datasets streaming shuffle: https://huggingface.co/docs/datasets/stream#shuffle
