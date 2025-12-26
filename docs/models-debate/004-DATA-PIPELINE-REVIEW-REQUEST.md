# Data Pipeline Review Request

This document is a review brief for external model feedback on the Phase 3
data pipeline (sources, packing, sizing, and assumptions).

Goal: confirm the pipeline is logically sound for our constraints and objectives,
and surface any missing checks or risks.

---

## Project Context (Short)

- Objective: demonstrate MoE expert specialization (code vs math vs prose),
  plus load balancing behavior and routing patterns. This is a demo/research
  project, not SOTA.
- Model: GPT-2 small (124M) with MoE layers in blocks 8-11.
- Routing: 8 experts, top-1, STE, load balancing loss, z-loss.
- Budget: $80 GPU total. Required runs: dense baseline, MoE main, no-LB ablation.

---

## What We Implemented

**Pipeline code:**
- `moe-emergence/data.py`

**Spec and decision docs:**
- `docs/DATA-PIPELINE.md`
- `docs/decisions/005-phase3-data-sizing.md`
- `docs/code-reviews/005-2025-12-26-data-py-fix.md`

**Current status:**
- Phase 3 pipeline implemented; data collection still pending
  (see `README.md` and `docs/README.md`).

---

## Data Sources (Current Plan)

**Code:**
- `codeparrot/codeparrot-clean` (train, streaming)
- Filter: 100 < len(text) < 10000 chars
- Note: multi-language, Python-heavy; no language filtering yet.

**Math:**
- `gsm8k` (main, train) + `hendrycks/competition_math` (train)
- Format: "Problem: ...\n\nSolution: ..."
- MATH only loads if GSM8K does not reach target size.
- Uses `trust_remote_code=True` for MATH.

**Prose:**
- `Salesforce/wikitext` (wikitext-103-raw-v1, train)
- Filter: 200 < len(text) < 10000 chars

---

## Packing and Sizing

- Sequence packing with EOS separators; no padding.
- Fixed block size: 512 tokens default (`--block-size` configurable).
- MB target is an approximation; actual size is reported in tokens/blocks.
- Warning if domain token counts differ by >1.5x.
- Optional `--balance-tokens`: truncate larger domains to smallest by block count.
- Tail tokens (remainder < block_size) are dropped during packing.

---

## Reproducibility

- Deterministic shuffle seed (`--seed`, default 42).
- Dataset metadata logged (dataset name/config/split, builder/version).
- Streaming data may change; versions are logged but not pinned.

---

## What We Want Feedback On

1. **Source selection**: Are these sources appropriate for the goal and budget?
   Any obvious better alternatives?
2. **Size target**: Is ~10MB per domain reasonable for specialization under
   this budget? Should we prefer more epochs vs more data?
3. **Domain balance**: Is the 1.5x warning + optional truncation sufficient?
   Should we balance by tokens, blocks, or something else?
4. **Language filtering**: Should code be Python-only given potential
   token-type analysis later, or is multi-language acceptable?
5. **Math mix**: Any concerns combining GSM8K + MATH (style mismatch,
   solution formats, LaTeX)?
6. **No train/val/test split**: Is this acceptable given the goal, or should
   we add a small held-out set for sanity checks?
7. **Packing choices**: Any pitfalls with EOS separators and tail-drop behavior?
8. **Evaluation risks**: Any missing checks that could invalidate the routing
   analysis or specialization claims?

---

## Reviewer Output Format (Preferred)

- **Findings** (ordered by severity)
- **Assumptions**
- **Recommendations**

If you suggest changes, please reference the file(s) above.
