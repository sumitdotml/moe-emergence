# Phase 5 Analysis Fix Log

- **Date:** 2026-02-26
- **Commit reference:** `bf277a61d5cb13a11faf3474d90d3fc3cec20972` (HEAD before committing this fix set)
- **Scope:** `scripts/export_wandb.py`, `moe_emergence/analysis.py`, `moe_emergence/visualize.py`, `notebooks/phase5_analysis.ipynb`, `notebooks/expertweights.ipynb`

## Issues Fixed

1. Optional W&B runs with missing router history could crash export verification.
2. Eval manifest metadata missed the required timestamp field from the Phase 5 plan contract.
3. Training-curves grouped bars omitted the no-lb ablation run.
4. The Phase 5 notebook was committed with heavy execution outputs.
5. Unrelated kernel metadata churn was present in `expertweights.ipynb`.

## Root Causes

1. `export_run()` used `Path()` as an empty sentinel and `main()` attempted `pd.read_csv(path)` when it resolved to the current directory.
2. Manifest serialization in `build_domain_eval_blocks()` omitted the timestamp key.
3. Bar-plot run filtering hard-coded only `dense`, `moe-main`, and `top-2`.
4. Notebook output/state was not cleared before staging.
5. Incidental metadata edits leaked into an unrelated notebook.

## Fixes Applied

1. Changed `export_run()` to return `None` on empty history and gated verification reads on `path is not None and path.is_file()`.
2. Added UTC ISO-8601 `timestamp` to eval manifest records and kept the hash mismatch warning behavior.
3. Included `no-lb` in grouped per-domain bars (with dynamic bar offsets/width) and preserved metric source-of-truth from `metrics.jsonl` best-eval rows.
4. Cleared code-cell outputs and execution counts in `notebooks/phase5_analysis.ipynb`.
5. Restored `notebooks/expertweights.ipynb` kernelspec display metadata to `.venv`.

## Visualization Quality Improvements

1. Added consistent axis styling helper for cleaner, more readable figures.
2. Switched domain heatmap to a true shared colorbar and added a uniform baseline note.
3. Added bar value labels in collapse comparison and highlighted the dominant expert trajectory line.
4. Added explicit marker overlay + cadence annotation (`logged every 100 steps`) in router entropy plot.
5. Added heatmap cell boundaries and tightened layout spacing for publication-ready output.

## Verification

1. `uv run ruff format moe_emergence/analysis.py moe_emergence/visualize.py scripts/export_wandb.py`
2. `uv run ruff check moe_emergence/analysis.py moe_emergence/visualize.py scripts/export_wandb.py`
3. `uv run python -m py_compile moe_emergence/analysis.py moe_emergence/visualize.py scripts/export_wandb.py`
4. Validated run selection logic includes all four runs (`dense`, `moe-main`, `no-lb`, `top-2`) when best eval rows exist.
5. Confirmed notebook hygiene: no execution outputs in `notebooks/phase5_analysis.ipynb` and expected kernelspec in `notebooks/expertweights.ipynb`.

## Related Checks

1. The W&B export script was not run end-to-end in this fix pass (network/API dependent), but the empty-history path now avoids local read failures.
2. Phase 5 chart functions remain aligned with `docs/project-design/PHASE-5-ANALYSIS-PLAN.md` figure definitions.

## Follow-Up Typing Compatibility Fixes (Same Session)

1. `moe_emergence/visualize.py`: changed `tight_layout(rect=...)` to a tuple to satisfy strict typing for the rect argument.
2. `moe_emergence/visualize.py`: replaced `plt.cm.tab10(...)` with `plt.get_cmap(\"tab10\")(... )` to avoid unresolved-attribute diagnostics.
3. `moe_emergence/analysis.py`: wrapped packed-block lists in a typed `Dataset` (`PackedBlocksDataset`) before passing to `DataLoader`.
4. `moe_emergence/analysis.py`: replaced `dict.update(util)` with explicit per-layer assignment in collapse trajectory to satisfy overload constraints for mixed key types.
5. Verification for follow-up fixes:
   - `uv run ruff check moe_emergence/analysis.py moe_emergence/visualize.py`
   - `uv run python -m py_compile moe_emergence/analysis.py moe_emergence/visualize.py`
   - `uv run ty check moe_emergence/analysis.py moe_emergence/visualize.py`
