# Phase 5: Post-Training Analysis & Visualization

## Context

Phase 4 training is complete (4 runs: dense baseline, MoE main, no-LB ablation, top-2 directional). We now need analysis code to produce publication-quality figures for the technical report. The V3 design doc's Phase 6/7 pseudocode has several discrepancies with the actual codebase — wrong function names, wrong dict keys, wrong tensor shapes, and missing router entropy data locally. This plan accounts for all of those gaps.

**Key discrepancies from V3 spec being corrected:**
- `collect_moe_aux()` → actual: `collect_aux_outputs()`
- `aux['indices']` → actual: `aux['topk_indices']`
- `aux['indices'][0, bpe_idx, 0]` (3D) → actual: `topk_indices[bpe_idx, 0]` (2D, batch×seq flattened)
- Router entropy not in `metrics.jsonl` — only in W&B
- `moe_modules` is `dict[int, MoEWrapper]` (V3 assumption correct, but pseudocode has bugs)

**Issues raised in cross-model review (`docs/models-debate/011-PHASE5-ANALYSIS-PLAN-CONVERGENCE-2026-02-26.md`):**
- [P1] Dataset reproducibility: streaming loaders aren't revision-pinned → add hash manifest
- [P1] pandas dependency missing from dep list → add explicitly
- [P1] Final sidecars for top-2/no-lb lack eval metrics → use metrics.jsonl eval rows
- [P2] Router entropy logged every 100 steps, not continuous → use markers, no interpolation
- [P2] Runtime estimate unverified → benchmark in notebook instead

---

## Files to Create

| File | Purpose |
|------|---------|
| `moe_emergence/analysis.py` | All computation: model loading, routing data collection, domain analysis, token-type analysis |
| `moe_emergence/visualize.py` | All plotting: heatmaps, bar charts, line plots. Returns `Figure` objects, no `plt.show()` |
| `scripts/export_wandb.py` | One-time W&B API export of router entropy to local CSV |
| `notebooks/phase5_analysis.ipynb` | Main notebook that ties analysis + visualization together |
| `figures/` | Output directory for saved figures (gitignored) |

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `matplotlib>=3.9`, `seaborn>=0.13`, `pandas>=2.0` to dependencies |
| `.gitignore` | Add `figures/` if not already present |

---

## Step 0: Dependencies

Add to `pyproject.toml` dependencies:
```
"matplotlib>=3.9",
"seaborn>=0.13",
"pandas>=2.0",
```
`pandas` is already transitively available via `wandb`, but we use it directly for DataFrame operations in `analysis.py` and the W&B export script, so it should be explicit.

Run `uv sync`.

---

## Step 1: `scripts/export_wandb.py` — W&B Router Metrics Export

One-time script to export router entropy + expert fraction data from W&B to local CSV. This is needed because `metrics.jsonl` only has loss/lr/throughput — router metrics were logged to W&B only via `tracking.py`.

**What it does:**
- Connects to W&B API (auth already configured)
- Exports `router/layer_{L}_entropy_per_token` and `router/layer_{L}_expert_{E}_frac` columns for MoE main run (`j08s2d1m`)
- Optionally exports for top-2 (`6mw6qbac`) and no-lb (`06pljhrv`)
- Saves to `.cache/wandb_exports/{run-name}-router-metrics.csv`

**Columns to export:**
```python
["_step"] + [f"router/layer_{l}_entropy_per_token" for l in [8,9,10,11]]
         + [f"router/layer_{l}_expert_{e}_frac" for l in [8,9,10,11] for e in range(8)]
         + ["router/entropy_per_token", "router/util_std"]
```

**Run IDs** (sourced from `checkpoints/README.md:174-177`):
- MoE main: `sumit-ml/moe-emergence/j08s2d1m`
- Top-2: `sumit-ml/moe-emergence/6mw6qbac`
- No-LB: `sumit-ml/moe-emergence/06pljhrv`

---

## Step 2: `moe_emergence/analysis.py`

### Reused existing functions (do NOT reimplement):
- `gpt2_moe.install_moe_layers()` — model setup
- `gpt2_moe.collect_aux_outputs()` — routing data extraction
- `data.load_code_data()`, `load_math_data()`, `load_prose_data()` — domain loading
- `data.split_texts_for_eval()` — eval split (must use same `seed=42` as training)
- `data.pack_sequences()` — sequence packing
- `data.collate_packed()` — batching

### Functions to implement:

**`load_moe_model(checkpoint_stem, device="cpu") -> (model, moe_modules, metadata)`**
- Reads `.json` sidecar for mode/config (follow pattern from `gpt2_inference.py:113-144`)
- Calls `install_moe_layers()` with config params if mode=="moe"
- Loads `.safetensors` weights
- Puts model in eval mode
- Returns empty dict for moe_modules if dense

**`build_domain_eval_blocks(domain, tokenizer, size_mb=10.0, block_size=512, seed=42) -> list[dict]`**
- Calls appropriate `load_*_data()` function
- Calls `split_texts_for_eval(texts, seed)` — returns `(train, eval)`, we keep eval
- Calls `pack_sequences(eval_texts, tokenizer, block_size, domain_label=domain)`
- Returns the packed blocks (list of `{"input_ids": Tensor, "domain": str}`)
- **Reproducibility guard**: After loading eval texts, compute `hashlib.sha256` over the concatenated eval texts and write a manifest to `.cache/eval_manifests/{domain}.json` containing `{"domain", "n_texts", "n_chars", "sha256", "timestamp", "size_mb", "seed"}`. On subsequent runs, if the manifest exists, verify the hash matches and warn (not error) on mismatch. The loaders for code (`codeparrot/codeparrot-clean`) and prose (`allenai/c4`) use `streaming=True` with no revision pin (`data.py:163, 324`). The iteration order is deterministic for a given HuggingFace dataset revision, but a future upstream re-upload could change it. MathQA loads from a pinned ZIP URL and is not at risk.

**`compute_domain_expert_fractions(model, moe_modules, domain_blocks, device, batch_size=4) -> dict`**
- For each domain, batches blocks using `collate_packed`, runs forward pass
- After each forward: `collect_aux_outputs(moe_modules)` → accumulate `topk_indices` via `torch.bincount(..., minlength=n_experts)`
- Returns `{layer_idx: {"code": Tensor[n_experts], "math": ..., "prose": ...}}` (normalized fracs)

**`compute_expert_utilization_at_snapshot(checkpoint_stem, all_blocks, device, batch_size=4) -> dict`**
- Loads model, runs all blocks, returns `{layer_idx: Tensor[n_experts]}` (aggregated across all domains)

**`compute_collapse_trajectory(snapshot_stems, all_blocks, device) -> list[dict]`**
- `snapshot_stems`: list of `(step, "path/stem")`
- Calls `compute_expert_utilization_at_snapshot` for each
- Returns `[{"step": 100, 8: Tensor[8], 9: Tensor[8], ...}, ...]`

**`compute_token_type_routing(model, moe_modules, code_texts, tokenizer, device, max_samples=200, max_seq_len=512) -> dict`**
- Processes each code sample individually (batch=1) for offset_mapping alignment
- `tokenizer(code, return_tensors='pt', return_offsets_mapping=True, truncation=True, max_length=max_seq_len)`
- Python `tokenize.generate_tokens()` on source (catch `TokenError`, skip bad files)
- Precompute `line_offsets` ONCE per sample (V3 had this inside the per-token loop — O(n²) bug)
- Map BPE tokens to Python token types via character offset overlap
- Distinguish `KEYWORD` from `NAME` using `keyword.iskeyword()`
- `topk_indices` shape is `[seq_len, topk]` for batch=1. Access: `topk_indices[bpe_idx, 0].item()`
- Returns `{layer_idx: {"KEYWORD": Tensor[8], "NAME": Tensor[8], "OP": ..., ...}}` (normalized)

**`load_metrics_jsonl(path) -> pd.DataFrame`**
- Parse metrics.jsonl, return DataFrame with all columns

**`get_best_eval_metrics(metrics_path) -> dict`**
- Parses `metrics.jsonl`, filters to rows containing `eval/loss`, returns the row with the lowest `eval/loss`. This is the correct source for "final eval" numbers — **do not use sidecar `.json` files** for this, because the final sidecars for top-2 and no-lb only store `train_loss`/`lm_loss` (no eval summary). The `best-model.json` sidecars do have eval, but using `metrics.jsonl` is more consistent across all runs.

---

## Step 3: `moe_emergence/visualize.py`

Style config: `dpi=150` display / `300` save, font 11pt, white background, no grid.
Domain colors: code=`#2196F3`, math=`#FF9800`, prose=`#4CAF50`.
Expert heatmap colormap: `YlOrRd`.

### Figures to implement:

**Fig 1-2: `plot_expert_domain_heatmap_grid(fractions, layers=[8,9,10,11])` → Figure**
- 2×2 subplot grid, one heatmap per layer
- Rows: Code/Math/Prose. Columns: E0–E7
- Cells annotated with 2-decimal fracs. `vmin=0, vmax=0.30` (uniform = 0.125)
- Shared colorbar. `figsize=(14, 10)`

**Fig 3: `plot_collapse_comparison(moe_fracs, nolb_fracs, layer_idx)` → Figure**
- Side-by-side bar charts: "With Load Balancing" vs "Without Load Balancing"
- Dashed line at 1/8 = 0.125 (uniform). `ylim=(0, 1.0)`
- `figsize=(10, 4)`

**Fig 4: `plot_collapse_trajectory(trajectory, layer_idx)` → Figure**
- Stacked area or multi-line showing expert fracs at steps 100→500 for one layer
- Shows dominant expert growing while others shrink
- `figsize=(10, 5)`

**Fig 5: `plot_router_entropy_over_training(entropy_df, layers=[8,9,10,11])` → Figure**
- Line plot, one line per layer. X=step, Y=mean entropy per token
- Dashed line at `ln(8) ≈ 2.079` (max entropy for 8 experts)
- Data from W&B export CSV
- **Sparse data**: router metrics were logged every 100 steps (`--log-router-every 100`, `train.py:1057`), not every step. Use scatter+line (markers at actual data points), no interpolation or smoothing. Caption should note the logging cadence.
- `figsize=(10, 5)`

**Fig 6: `plot_token_type_routing(token_type_fracs, layer_idx)` → Figure**
- Heatmap: rows=token types (KEYWORD, NAME, NUMBER, STRING, OP, COMMENT, OTHER), cols=E0–E7
- Same style as domain heatmap. `figsize=(10, 5)`

**Fig 7: `plot_training_curves(metrics_dict)` → Figure**
- Two panels: (left) eval/loss over steps for all 4 runs, (right) per-domain best-eval loss grouped bars
- Per-domain values sourced from `get_best_eval_metrics()` (lowest `eval/loss` row in `metrics.jsonl`), not from sidecar JSON
- `figsize=(12, 5)`

All functions accept optional `save_path` to save directly.

---

## Step 4: `notebooks/phase5_analysis.ipynb`

Cell order:
1. **Setup** — imports, paths, style config, device selection (MPS if available, else CPU)
2. **Build eval data** — tokenizer + 3 domain eval block sets (~2 min with data download)
3. **Expert-domain heatmaps** — load moe-main final, compute fractions, plot grid
4. **Collapse comparison** — load no-lb snapshots (steps 100-500), compute trajectory, compare with moe-main
5. **Router entropy** — load W&B CSV, plot per-layer entropy over training
6. **Token-type routing** — analyze code eval texts through moe-main model, plot
7. **Training curves** — parse all 4 `metrics.jsonl` files, plot
8. **Save all figures** — batch save to `figures/`

---

## Execution & Verification

1. `uv sync` after adding deps
2. `uv run python scripts/export_wandb.py` — verify CSV files created in `.cache/wandb_exports/`
3. Run notebook cells sequentially — each cell should produce a visible figure
4. Verify `figures/` contains all saved PNGs
5. Spot-check: moe-main heatmap should show non-uniform expert utilization across domains; no-lb collapse should show one expert dominating (~73% at step 500 in layer 9); router entropy should decrease over training

**Runtime**: Not estimated upfront. The notebook includes a benchmark cell (cell 2) that times eval data construction and a single model load + forward pass, then extrapolates total time for the full analysis. Report device (CPU vs MPS) and actual elapsed time per section.

---

## Snapshot Inventory for Reference

| Run | Snapshots Available | Use |
|-----|-------------------|-----|
| moe-main | `final-model` only | Domain fracs, token-type routing, collapse comparison (balanced side) |
| no-lb | `model-step-{100,200,300,400}` + `final-model` (step 500) | Collapse trajectory, collapse comparison (collapsed side) |
| top-2 | `model-step-{1000,3000,5000,8000}` + `final-model` | Optional: compare top-2 routing patterns with top-1 |
| dense | `final-model` only | Training curves only (no routing data) |
