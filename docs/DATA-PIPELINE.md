# Data Pipeline

This document describes the data pipeline for training the MoE model on three distinct domains: code, math, and prose.

## Overview

The pipeline creates a packed, multi-domain dataset for efficient causal language model training. Key characteristics:

- **No padding**: Sequences are packed end-to-end with EOS separators
- **Fixed block size**: All training samples are exactly `block_size` tokens (default 512)
- **Domain labels preserved**: Each block is tagged with its source domain for post-hoc analysis
- **Token-based reporting**: MB targets are approximate; actual sizing is *measured* in tokens (not sized by tokens)
- **No train/val/test split**: All data is used for training; evaluation is separate

## Data Sources

### Code: CodeParrot

| Property | Value |
|----------|-------|
| Dataset | `codeparrot/codeparrot-clean` |
| Split | `train` |
| Content | Code from GitHub (multi-language, Python-heavy) |
| Streaming | Yes |

**Filtering:**
- Minimum length: 100 characters (hardcoded default)
- Maximum length: 10,000 characters (configurable via `--max-example-chars`)

**Why CodeParrot:**
- Large, diverse Python codebase
- Clean version removes duplicates and low-quality files
- Streaming support for memory efficiency

### Math: GSM8K + MATH

| Property | GSM8K | MATH |
|----------|-------|------|
| Dataset | `gsm8k` (config: `main`) | `hendrycks/competition_math` |
| Split | `train` | `train` |
| Content | Grade school math problems | Competition-level math |
| Size | ~7,500 examples | ~12,500 examples |

**Format:**
```
Problem: {problem_text}

Solution: {solution_text}
```

**Loading order:**
1. GSM8K is loaded first (smaller, faster)
2. MATH is loaded only if GSM8K doesn't reach the target size
3. Both datasets are combined into a single math domain

**Why two datasets:**
- GSM8K alone is ~2-4MB, insufficient for 10MB target
- MATH provides additional problems with LaTeX notation
- Combined, they provide sufficient diversity

### Prose: WikiText-103

| Property | Value |
|----------|-------|
| Dataset | `Salesforce/wikitext` |
| Config | `wikitext-103-raw-v1` |
| Split | `train` |
| Content | Wikipedia articles |

**Filtering:**
- Minimum length: 200 characters (hardcoded default)
- Maximum length: 10,000 characters (configurable via `--max-example-chars`)

**Why WikiText-103:**
- OpenWebText has compatibility issues with newer `datasets` library
- WikiText-103 is well-maintained and widely used
- Raw version preserves natural text structure

## Sequence Packing

Instead of padding each text to a fixed length (wasteful), we use sequence packing:

```
Text 1 tokens | EOS | Text 2 tokens | EOS | Text 3 tokens | EOS | ...
|<----- block 1 ----->|<----- block 2 ----->|<----- block 3 ----->|
```

**Algorithm:**
1. Tokenize all texts in a domain
2. Concatenate with EOS token separators
3. Chunk into fixed-size blocks
4. Each block may contain parts of multiple documents

**Note:** Tail tokens (remainder < block_size) are dropped. Reported token counts include all tokens before chunking, so actual tokens used in training may be slightly fewer (at most `block_size - 1` per domain).

**Benefits:**
- No wasted computation on padding tokens
- No need for loss masking
- Standard practice for causal LM pretraining

**Implementation:** `pack_sequences()` in `data.py`

## Token Balancing

### The Problem

Equal MB does not mean equal tokens:
- Code tokenizes densely (variable names, syntax → more tokens per char)
- Prose tokenizes normally
- Math varies (LaTeX can be verbose)

This can skew domain representation by 1.5-2x.

### The Solution

1. **Always report token counts** per domain (authoritative metric)
2. **Warn if imbalance >1.5x** ratio between smallest and largest domain
3. **Optional truncation** via `--balance-tokens` flag

**Truncation strategy:**
- Truncate larger domains to match the smallest (by block count)
- No oversampling (risks inflating specialization via repetition)

**Example output:**
```
Warning: Token imbalance detected (>1.5x ratio)!
  Smallest: prose with 112,650 tokens
  Largest has 223,025 tokens

Balanced to 440 blocks per domain (1320 total, ~337,920 tokens)
```

## Reproducibility

### Seeds

- Shuffle seed is configurable via `--seed` (default: 42)
- Same seed + same data = same block order

### Dataset Metadata

Each loader returns metadata including:
- Dataset name and split
- Builder name and version (from HuggingFace, may vary by dataset format)
- Number of examples and total characters
- Filtering parameters used

**Example:**
```python
{
    'dataset': 'codeparrot/codeparrot-clean',
    'split': 'train',
    'builder_name': 'json',  # may vary: 'json', 'parquet', etc.
    'version': '0.0.0',      # may be absent for some datasets
    'num_examples': 132,
    'total_chars': 528263,
    'min_example_chars': 100,
    'max_example_chars': 10000
}
```

### Caveats

- Streaming datasets may yield different samples if upstream data changes
- Dataset versions are logged but not pinned
- For strict reproducibility, cache the loaded texts locally

## CLI Reference

```bash
# Quick test (small dataset)
uv run python moe_emergence/data.py --size-mb 1 --block-size 256

# Development iteration
uv run python moe_emergence/data.py --size-mb 5

# Final runs with balancing
uv run python moe_emergence/data.py --size-mb 10 --balance-tokens --seed 42
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--size-mb` | 10 | Target size per domain in MB |
| `--block-size` | 512 | Tokens per training block |
| `--balance-tokens` | False | Truncate larger domains to match smallest |
| `--seed` | 42 | Random seed for shuffling |
| `--max-example-chars` | 10000 | Per-example length cap |

## Known Limitations

1. **Token counting is post-hoc**: We load by character count, then measure tokens. This means the actual token count may differ from the MB target.

2. **No cross-domain blocks**: Each block contains text from only one domain. This simplifies domain labeling but may not be optimal for learning.

3. **WikiText vs OpenWebText**: WikiText-103 is more formal (Wikipedia) than OpenWebText (Reddit/web). This may affect prose style diversity.

4. **MATH dataset requires `trust_remote_code=True`**: The MATH dataset uses a custom loading script.

5. **No deduplication across domains**: A code snippet discussing math or a Wikipedia article about programming would be in separate domains.

## File Structure

```
moe_emergence/data.py
├── _dataset_meta()           # Extract HF dataset metadata
├── pack_sequences()          # Core packing logic
├── load_code_data()          # CodeParrot loader
├── load_math_data()          # GSM8K + MATH loader
├── load_prose_data()         # WikiText loader
├── PackedMixedDomainDataset  # PyTorch Dataset class
├── collate_packed()          # DataLoader collate function
└── main()                    # CLI entry point
```

## References

- Decision doc: `docs/decisions/005-phase3-data-sizing.md`
- Design spec: `docs/project-design/MOE-PROJECT-DESIGN-V3.md` (lines 922-1109)
- Code review: `docs/code-reviews/005-2025-12-26-data-py-fix.md`
