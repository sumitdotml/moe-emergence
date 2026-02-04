# Decision 011: Formatting Artifact Verification

**Date**: 2026-02-04
**Status**: Verified
**Commit**: 326780b (verification script added this session)

## Context

MMLU-Pro had a data leakage bug where correct answers disproportionately started with leading whitespace. This was invisible in the HuggingFace viewer but detectable via `load_dataset()`. Models could exploit this formatting artifact instead of learning content. The issue affected Physics, Math, and Chemistry categories.

Reference: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41

We needed to verify our datasets (CodeParrot, MathQA, C4) don't have similar issues that could create routing shortcuts for the MoE router.

## Verification Performed

Created `moe_emergence/verify_formatting.py` which checks:
1. Leading/trailing whitespace patterns
2. Invisible Unicode characters (BOM, zero-width spaces, etc.)
3. First character distribution (for systematic bias)
4. Unicode category analysis

Ran on 1000 samples per domain.

## Results

| Domain | Leading Whitespace | Invisible Chars | First Char Bias |
|--------|-------------------|-----------------|-----------------|
| CODE   | 0%                | 7 (0.7%)        | 58% `#` (expected for Python) |
| MATH   | 0%                | 4 (0.4%)        | Normal distribution |
| PROSE  | 0%                | 10 (1.0%)       | 15% `T` (normal English) |

**Key findings:**
- No MMLU-Pro-style systematic leading whitespace bias
- Invisible characters exist but at <1% rate, randomly distributed
- CODE starting with `#` is a legitimate Python signal (shebang, comments), not an artifact
- MATH and PROSE have natural first-character distributions

## Decision

**Datasets are clean for training.** No remediation needed.

The rare invisible characters (BOM, zero-width space) are scattered randomly and won't create exploitable routing shortcuts. They're not correlated with any domain signal.

## Alternatives Considered

1. **Strip all invisible characters** — Rejected. Low occurrence rate (<1%), no systematic bias, adds preprocessing complexity for no benefit.

2. **Filter samples with invisible chars** — Rejected. Would lose valid samples for a non-issue.

## Implications

- Proceed to training without data cleaning
- The verification script is available for future dataset additions
- No need to re-run unless datasets change
