# Decision: Final Dataset Choices for 3-Domain Training

**Date:** 2025-01-28
**Status:** Accepted
**Context Commit:** `4b29ec4`

---

## Context

Need to select datasets for the three training domains (code, math, prose) that:
1. Have clearly distinct content styles for expert specialization
2. Are accessible and properly licensed
3. Are appropriately sized for our $80 GPU budget

---

## Options Considered

### Code Domain

| Dataset | Pros | Cons |
|---------|------|------|
| **CodeParrot-clean** | Diverse repos, multiple licenses, real production Python | — |
| StarCoderData | Larger, higher quality | Requires TOS acceptance |

### Math Domain

| Dataset | Pros | Cons |
|---------|------|------|
| **MathQA (allenai)** | 29K word problems with rationales, Apache 2.0 | — |

### Prose Domain

| Dataset | Pros | Cons |
|---------|------|------|
| WikiText-103 | Clean, structured | Too formal, encyclopedia-style, has `@-@` artifacts |
| OpenWebText | Conversational | News/articles could overlap with code comments |
| **C4 (allenai/c4, en)** | Natural web text, well-filtered, diverse | Large (but streaming works) |
| FineWeb | High quality, curated | Less natural than C4 based on sample review |
| RedPajama-V2 | State-of-art curation | Complex access, requires S3 |

---

## Decision

| Domain | Dataset | Rationale |
|--------|---------|-----------|
| **Code** | `codeparrot/codeparrot-clean` | Diverse Python code, easy access, good licenses |
| **Math** | MathQA (from ZIP) | Word problems with step-by-step rationales |
| **Prose** | `allenai/c4` (en) | Most natural-sounding web text after sample comparison |

### Why C4 over alternatives

After generating 100 random samples from both C4 and FineWeb, C4 showed more natural, conversational prose. WikiText-103 was rejected for being too formal (encyclopedia-style). OpenWebText was rejected because news/article style could overlap with code documentation.

---

## Consequences

**Positive:**
- Three clearly distinct content styles for expert specialization
- All datasets accessible via HuggingFace streaming
- Well-documented, commonly-used datasets

**Negative:**
- C4 is large (750GB total) but streaming mitigates this
- Some noise in web-crawled data (acceptable for this project)

**Implementation:**
- Code: `load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)`
- Math: Custom loader from MathQA ZIP (already implemented in `data.py`)
- Prose: `load_dataset("allenai/c4", "en", split="train", streaming=True)`

---

## References

- Sample comparison files: `samples_c4_allenai.txt`, `samples_fineweb_hf.txt`
- C4 dataset: https://huggingface.co/datasets/allenai/c4
- CodeParrot-clean: https://huggingface.co/datasets/codeparrot/codeparrot-clean
- MathQA decision: `docs/decisions/007-math-prefix-randomization.md`
