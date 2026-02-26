# Model Debate: Phase 5 Analysis Plan Review

This document contains a structured debate between AI models to create the optimal plan.

---

## Claude Opus 4.6 - Analysis 1

### Understanding the Request
Review `docs/project-design/PHASE-5-ANALYSIS-PLAN.md` with strict, implementation-grounded feedback, then propose a stronger Phase 5 execution plan for post-training analysis and publication figures.

Baseline commit for this review: `2924c126ad6c02ef1dc53b02a3b5e07afa903660`.

### Key Constraints and Considerations
- Existing implementation contracts are fixed and must be reused (`collect_aux_outputs`, `topk_indices`, `install_moe_layers`).
- Checkpoint inventory is fixed: `moe-main` has final only, `no-lb-ablation` has 100/200/300/400/final, `top2-main-10k` has 1000/3000/5000/8000/final.
- Router metrics are not in local `metrics.jsonl`; they are in W&B logs.
- Data loaders for code/prose use streaming sources, which can drift over time if not pinned and snapshotted.
- Current dependency list does not include plotting/notebook stack required by the proposed notebook workflow.

### Proposed Approach
The current Phase 5 plan is mostly directionally correct. The critical adjustments needed are:
1. Add a reproducibility layer (dataset manifest plus checkpoint and metric source manifest) before analysis.
2. Correct dependency and environment assumptions (pandas, notebook kernel, and explicit W&B export preflight).
3. Replace optimistic runtime claims with measured local benchmarking and clear fallback options.
4. Add statistical rigor for claims (confidence intervals and null/permutation checks).
5. Tighten metric semantics (step alignment, sparse router logs, and eval selection rules for cross-run comparison).

### Trade-offs and Decisions Made
- I favor stronger reproducibility and stats gates over a faster but fragile notebook-only path.
- I keep the modular split (`analysis.py` and `visualize.py`) since it improves testability and re-run cost.
- I recommend retaining W&B export as an ingestion step, but with robust handling for sparse/missing columns.

### Potential Weaknesses
- If the original exact training eval text set was not persisted, strict train-eval reconstruction cannot be guaranteed for streaming datasets.
- Statistical tests add implementation time and may delay figure generation by one iteration.
- Runtime on CPU can vary widely by machine; benchmark gates need to be calibrated locally.

### Questions for GPT 5.3 Codex
1. Do you agree that dataset drift from streaming sources is the highest-risk validity issue for Phase 5 domain routing claims?
2. Is bootstrap CI + permutation test sufficient, or should we require an additional effect-size threshold per figure?
3. For cross-run training curves, should we compare best eval point per run, fixed-step checkpoints, or both?
4. Should we include top-2 analyses in the first Phase 5 deliverable, or keep them explicitly optional to reduce scope?

---

## GPT 5.3 Codex - Analysis 1

### Understanding the Request
Review whether the updated `docs/project-design/PHASE-5-ANALYSIS-PLAN.md` actually resolves the five prior findings without introducing new execution ambiguity.

### Key Constraints and Considerations
- The plan must align with current training artifacts and code contracts already present in repo.
- Changes should preserve feasibility for Phase 5 delivery (not explode scope).
- Fixes should be explicit enough that implementation does not depend on hidden assumptions.

### Proposed Approach
I evaluated each claimed fix against the updated plan text and existing repository structure.

Result: the five requested items are materially addressed, with two residual clarity gaps that should be patched before implementation starts.

### Trade-offs and Decisions Made
- I am treating this pass as a validation review, not a redesign pass.
- I am not reopening broader scope items (e.g., full inferential statistics package) because they were not part of the five-item correction request.
- I prioritize implementation clarity issues that can cause drift or confusion during execution.

### Potential Weaknesses
1. **Dependency consistency gap (minor but real):** `Step 0` adds `pandas`, but the `Files to Modify` table still lists only matplotlib/seaborn. This is an avoidable inconsistency.
2. **Reproducibility metadata still light:** hashing eval texts is a good guard, but plan text does not yet require saving a structured manifest path (dataset id/revision, seed, hash, counts) for auditability.
3. **Run-ID provenance wording:** run IDs are still labeled “from memory”; they should be sourced from a stable local doc (`checkpoints/README.md`) to avoid accidental mismatch.

### Answers to Claude’s Questions
1. Yes. Dataset drift from streaming sources is still the highest validity risk for domain-routing claims, and the new hash guard is the right first mitigation.
2. Bootstrap CI + permutation test is sufficient for this phase. Add effect-size thresholds only if you see unstable or borderline effects.
3. Use both: full eval trajectories plus a clearly marked best-eval point per run.
4. Keep top-2 as optional in first deliverable; include only if core figures are complete and reproducible.

### Questions for Claude Opus 4.5
1. Can we add a single mandatory manifest output path (for example `.cache/phase5/analysis_manifest.json`) to lock reproducibility metadata?
2. Should we normalize the “Files to Modify” table to match Step 0 exactly (`pandas` included) to prevent implementation drift?
3. Do you agree we should replace “Run IDs (from memory)” with “Run IDs from checkpoints/README.md”?

---

## Claude Opus 4.6 - Response 1

### Answers to GPT 5.3 Codex's Questions

1. **Mandatory manifest path**: Yes. Adopted `.cache/eval_manifests/{domain}.json` with schema `{"domain", "n_texts", "n_chars", "sha256", "timestamp", "size_mb", "seed"}`. Written on first load, verified on subsequent runs (warn on mismatch, not error — since the streaming sources may have legitimate upstream updates).

2. **Normalize Files to Modify table**: Yes. Updated to include `pandas>=2.0` alongside matplotlib/seaborn. Now matches Step 0 exactly.

3. **Run ID provenance**: Yes. Changed from "from memory" to "sourced from `checkpoints/README.md:174-177`". Verified the README contains the canonical W&B links for all four runs.

### Changes Applied to Plan

All three items applied to `docs/project-design/PHASE-5-ANALYSIS-PLAN.md` in the same session:

| GPT 5.3 Codex Finding | Fix | Location in Plan |
|---|---|---|
| Dependency table inconsistency | Added `pandas>=2.0` to Files to Modify table | Line 37 |
| Reproducibility manifest unstructured | Defined manifest path `.cache/eval_manifests/{domain}.json` with explicit schema | Lines 104–105 |
| Run IDs "from memory" | Changed to "sourced from `checkpoints/README.md:174-177`" | Line 73 |

### Assessment

GPT 5.3 Codex's validation pass was precise — all three residual items were real and actionable. No disagreements. The plan is now internally consistent across the dependency list, reproducibility layer, and provenance references.

---

## GPT 5.3 Codex - Validation 1

No critical findings. Claude's three claimed fixes are present in the updated plan:

1. Dependency table now includes `pandas>=2.0` in the "Files to Modify" table at `PHASE-5-ANALYSIS-PLAN.md:37`.
2. Reproducibility manifest path/schema is defined at `PHASE-5-ANALYSIS-PLAN.md:104`.
3. Run IDs are now explicitly sourced from README at `PHASE-5-ANALYSIS-PLAN.md:73`.

The canonical implementation plan remains `docs/project-design/PHASE-5-ANALYSIS-PLAN.md`.

**Residual risk only**: current spec says "warn on manifest mismatch" (not fail), which is fine for exploratory analysis but weaker for strict reproducibility gates.

---

## Convergence Summary

**Status: Converged.**

Both models agree on:
- The five original P1/P2 items are resolved.
- Three residual clarity gaps identified by GPT 5.3 Codex (dependency table, manifest schema, run ID provenance) are now patched.
- Top-2 analysis remains optional in first deliverable.
- Bootstrap CI + permutation tests deferred unless borderline effects appear.
- Manifest mismatch behavior (warn vs fail) is acceptable for exploratory Phase 5 analysis; can be tightened to fail-on-mismatch if results are used for publication claims.
- Plan is ready for implementation.
