# Decision: Phase 3 Dataset Sizing and Token-Based Balancing

**Date:** 2025-12-26
**Status:** Accepted
**Context Commit:** `07c29ba`

---

## Context

Phase 3 requires implementing `data.py` with sequence packing and multi-domain datasets (code, math, prose). The V3 design spec uses MB-based sizing (10MB per domain), but MB doesn't map cleanly to tokens—code/math/prose tokenize differently. This decision documents the refined approach developed through Opus 4.5 + GPT-5.2 discussion.

---

## Options Considered

### Option A: MB-only sizing (V3 default)

**Description:** Use character-count-based MB targets as specified in V3 design.

**Pros:**

- Simple to implement
- Matches existing design doc

**Cons:**

- Equal MB ≠ equal tokens (code tokenizes denser than prose)
- Can skew domain balance by 1.5-2x
- May confound "specialization" with exposure imbalance

### Option B: Token-based sizing with reporting

**Description:** Keep MB as a convenience target, but report and optionally balance by token counts.

**Pros:**

- More accurate domain balance
- Transparent sizing (authoritative metric is tokens/blocks)
- Optional balancing via truncation keeps it simple

**Cons:**

- Slightly more complex implementation
- Truncation can shrink dataset below MB target

---

## Decision

Choose **Option B**: Token-based reporting with optional balancing.

**Key decisions:**

- **Token-based reporting**: Always report tokens per domain (not just MB)
- **Imbalance warning**: Warn if any domain differs by >1.5x tokens from others
- **Optional balancing**: `--balance-tokens` flag (off by default)
  - Strategy: Truncate larger domains to match smallest (by token count, not chars)
  - Rationale: Oversampling risks inflating specialization by repeating examples
- **Math sources**: GSM8K + MATH combined (GSM8K alone ~2-4MB, too small for 10MB target)
- **Reproducibility**: Deterministic shuffle seed + dataset split/version logging
- **Block size**: Configurable (default 512)
- **Per-example length cap**: Filter out extremely long samples (>10K chars)

---

## Consequences

- **Positive:**
  - More accurate domain balance for fair specialization analysis
  - Transparent sizing—users see actual tokens/blocks, not just MB proxy
  - Reproducible runs via seed and dataset version logging
  - Flexible iteration via CLI flags (size, block-size, balancing)

- **Negative:**
  - Truncation can shrink dataset below target if one domain is small
  - Slightly more output to parse (token counts, warnings)

- **Risks:**
  - Streaming + filtering may yield non-deterministic samples if upstream data changes; logging dataset versions/configs helps track drift

---

## Implementation Plan

### Components to Implement

**1. `pack_sequences()` function**

- Tokenizes texts and concatenates with EOS separators
- Chunks into fixed-size blocks (default 512 tokens)
- Returns list of `{'input_ids': Tensor, 'domain': str}`
- No padding tokens

**2. Data Loading Functions**

| Function            | Dataset(s)                                                   | Notes                           |
| ------------------- | ------------------------------------------------------------ | ------------------------------- |
| `load_code_data()`  | `codeparrot/codeparrot-clean`                                | Filter: 100 < len < 10000 chars |
| `load_math_data()`  | `gsm8k` (main, train) + `hendrycks/competition_math` (train) | Combined; consistent format     |
| `load_prose_data()` | `Salesforce/wikitext` (wikitext-103-raw-v1)                  | Filter: len > 200 chars         |

**Math formatting** (consistent across GSM8K and MATH):

```
Problem: {problem_text}

Solution: {solution_text}
```

**3. `PackedMixedDomainDataset` class**

- Packs each domain separately (preserving labels)
- Reports token counts per domain
- Warns if imbalance >1.5x
- Optional: truncate to balance tokens
- Shuffles all blocks together

**4. `collate_packed()` function**

- Stacks input_ids tensors
- Preserves domain labels for routing analysis

### File Structure

```
moe-emergence/data.py
├── pack_sequences()
├── load_code_data(max_size_mb)
├── load_math_data(max_size_mb)
├── load_prose_data(max_size_mb)
├── PackedMixedDomainDataset
├── collate_packed()
└── if __name__ == "__main__":  # verification with token reporting
```

### CLI Flags

- `--size-mb`: Target size per domain in MB (default: 10)
- `--block-size`: Tokens per block (default: 512)
- `--balance-tokens`: Truncate larger domains to match smallest
- `--seed`: Random seed for shuffle (default: 42)
- `--max-example-chars`: Per-example length cap (default: 10000)

### Verification & Output

- All blocks exactly `block_size` tokens
- No padding tokens present
- Token counts reported per domain (authoritative size metric)
- Block counts reported (for step estimation)
- Warning if imbalance >1.5x (includes smallest domain count for context)
- If balancing applied: Log post-balance size explicitly
- Dataset info logged: Split names, dataset versions/revisions

---

## References

- **Data pipeline spec**: `docs/DATA-PIPELINE.md`
- Design spec: `docs/project-design/MOE-PROJECT-DESIGN-V3.md` (lines 922-1109)
- Code review: `docs/code-reviews/005-2025-12-26-data-py-fix.md`
