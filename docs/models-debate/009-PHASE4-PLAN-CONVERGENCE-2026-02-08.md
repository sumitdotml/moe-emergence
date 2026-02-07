# Model Debate: Phase 4 Training Plan Critical Review

This document contains a structured debate between AI models to critically evaluate the Phase 4 Training Plan produced by GPT 5.3 Codex (`docs/project-design/PHASE-4-TRAINING-PLAN.md`).

---

## Claude Opus 4.6 — Analysis 1

### Understanding the Request

GPT 5.3 Codex produced a Phase 4 training plan that defines CLI interface, preset configurations, data flow, loss computation, tracking integration, checkpointing/resume, collapse detection, and mandatory shakedown gates. I'm evaluating this plan against: (a) the V3 design spec, (b) the actual codebase, and (c) implementation feasibility.

### Assessment: What the Plan Gets Right

**Strengths — these are solid and well-considered:**

1. **Single-script dual-mode (dense + MoE)** — Correct architecture. Avoids code duplication and makes A/B comparison trivial. V3 spec implies this separation but the plan handles it more cleanly.

2. **Step-based loop with iterator reset** — Better than V3's epoch-based approach. Packed datasets have variable block counts depending on token balancing; step-based gives deterministic run lengths across configurations.

3. **Preset system** — Good ergonomics. Encapsulating `{dense, moe-main, no-lb, top2, shakedown}` configs avoids user error from juggling 15+ flags.

4. **Dual checkpoint format (full-resume + model-only)** — Practical. Full checkpoints are large (optimizer state doubles size); model-only snapshots are useful for inference and analysis without the overhead. V3 only specifies full checkpoints.

5. **`--keep-last-k` retention** — V3 doesn't mention checkpoint rotation. At ~500MB per full checkpoint with 8 experts, keeping all checkpoints would burn disk fast. Good addition.

6. **3-consecutive-checks collapse criterion** — V3 uses single-check (`>60%` → stop). The plan's 3-check window reduces false triggers from batch-level variance. Materially better.

7. **Local JSONL/CSV metric log** — Safety net for W&B failures. V3 doesn't mention this. Practical.

8. **Shakedown gate as mandatory prerequisite** — Good operational discipline. V3 has `verify_moe_installation()` as a code function but doesn't enforce it as a gate before budgeted runs.

### Issues Found

#### ISSUE 1: Batch size mismatch — MEDIUM

| Plan | V3 Spec |
|------|---------|
| `batch_size=2, grad_accum=4` (effective 8) | `batch_size=8` (no grad accum mentioned) |

The plan arrives at the same effective batch size (8) via gradient accumulation, which is memory-safe for MPS/consumer GPUs. This is actually *better* than V3 for the stated dual-device target. However:

- The plan doesn't document *why* `batch_size=2` was chosen or what memory constraint it targets.
- No fallback: if even batch_size=2 OOMs on MPS (GPT-2 + 8 experts × 4 layers is nontrivial), there's no guidance.

**Recommendation:** Accept the plan's `2×4` over V3's `8×1`, but add a note about expected peak memory and what to do if MPS OOMs (reduce to `batch_size=1, grad_accum=8`).

#### ISSUE 2: Epoch vs. step scheduler mismatch — LOW

V3 computes `total_steps = len(train_loader) * config.epochs`. The plan uses `--max-steps` directly. These are functionally equivalent when `max_steps` matches the intended training budget, but:

- The plan doesn't specify how `total_steps` for the cosine scheduler is computed. If `max_steps=10000` and the scheduler uses the same value, the cosine decays to 0 at exactly step 10000. Correct.
- But if the user overrides `--max-steps` without adjusting the scheduler, the LR schedule could be misaligned. The plan should explicitly state: **scheduler total_steps = max_steps**.

**Recommendation:** Add explicit note: scheduler `num_training_steps` is always set to `max_steps` (not derived from dataset size).

#### ISSUE 3: Noise annealing configuration missing from presets — MEDIUM

V3 spec calls `set_anneal_steps(total_steps, anneal_fraction=0.25)` on each router. The plan's preset definitions don't include `anneal_fraction` as a configurable parameter, nor does the CLI expose it. It's presumably hardcoded at 0.25.

This matters because:
- The shakedown preset (100 steps) with 25% annealing = 25 steps of noise, which is fine.
- But the no-lb preset (2000 steps max, could early-stop at ~500) with 25% = 500 steps of noise — that's the entire run before collapse detection kicks in. The router never reaches zero noise before collapse is measured.

**Recommendation:** Either (a) expose `--anneal-fraction` in CLI, or (b) document the hardcoded 0.25 choice and verify it's appropriate for each preset. For `no-lb`, consider whether noise should even be enabled (if we're testing collapse without load balancing, adding exploration noise confounds the result).

#### ISSUE 4: Missing `--anneal-fraction` and `--noise-std` from CLI — LOW

Related to above but separable: the Router's `noise_std` parameter (default 0.1 in `install_moe_layers`) is not exposed in the plan's CLI. While defaults are probably fine, these are knobs an experimenter might want to tune.

**Recommendation:** Low priority. Accept defaults but document them in preset definitions.

#### ISSUE 5: `format_version` in checkpoints — undefined — LOW

The plan specifies `format_version` as a required checkpoint key but doesn't define what value it should have or what compatibility checks use it. This is forward-looking infrastructure with no current consumer.

**Recommendation:** Set `format_version=1` and check it on resume. Skip versioned migration logic until there's an actual format change.

#### ISSUE 6: V3 spec uses epoch-based training, plan uses step-based — clarification needed — LOW

V3 spec: `epochs: int = 3` in TrainingConfig. Plan: no epoch concept, pure step-based with iterator reset.

These are *not* equivalent when the dataset is small. With 10MB per domain, balanced tokens, block_size=512:
- Approximate blocks per domain: ~4000 (10MB ÷ ~2.5 bytes/token ÷ 512 tokens/block × token efficiency)
- Total training blocks: ~12000
- At batch_size=2, grad_accum=4: effective batch = 8, so ~1500 steps per "epoch"
- 3 epochs = ~4500 steps, which is *less* than the plan's 5000 for dense or 10000 for MoE

The plan's step counts imply multiple passes over the data. This is fine — expected for small datasets — but should be documented to avoid confusion about "epoch" semantics.

**Recommendation:** Document expected epochs-equivalent for each preset (e.g., "moe-main: ~6.7 epochs over the dataset").

#### ISSUE 7: `--balance-tokens` default in presets — inconsistency — LOW

The plan says `--balance-tokens` is a flag with "preset" as default, and the shakedown preset sets `balance_tokens=false`. But for training presets (dense, moe-main, no-lb, top2), the plan doesn't explicitly state the default. CLAUDE.md says `--balance-tokens required for training`.

**Recommendation:** Explicitly set `balance_tokens=true` in all non-shakedown presets.

#### ISSUE 8: `best-model.pt` selection criteria — undefined — MEDIUM

The plan says "Always retain `best-model.pt`" but doesn't define what "best" means. Lowest eval loss? Lowest per-domain eval loss? Lowest total loss? When is it updated?

**Recommendation:** Define `best-model.pt` as the checkpoint with the lowest overall eval loss (not per-domain). Update at each eval step if current eval loss < best seen. Save as model-only snapshot.

#### ISSUE 9: Data flow section — incomplete split handling — LOW

The plan says "Build tokenizer and packed train/eval datasets via data.py" but doesn't mention `split_texts_for_eval()` or `compute_eval_count()`. The actual data module splits texts *before* packing, which is important because:
- Eval blocks must be from held-out texts, not random packed blocks from training
- The `balance_tokens=False` flag on eval dataset is a deliberate choice

**Recommendation:** Mention text-level split explicitly. Reference `split_texts_for_eval()`.

#### ISSUE 10: `--size-mb` default is 10.0 — but shakedown uses 1.0 — fine but note

The plan correctly overrides `size_mb=1` for shakedown. But the CLI default is 10.0, and presets override this. No issue — just confirming the override chain works.

#### ISSUE 11: Missing `--max-example-chars` from CLI — LOW

The data module supports `--max-example-chars` (default 10000) for per-example length caps. The plan's CLI doesn't expose this. Probably fine for default usage.

### What's Not in the Plan but Should Be Considered

1. **Reproducibility**: The plan specifies `--seed 42` but doesn't mention setting `torch.manual_seed`, `torch.cuda.manual_seed_all`, `random.seed`, `np.random.seed` — these are all necessary for full reproducibility and checkpoint RNG state is listed in checkpoint keys but the actual seeding protocol isn't specified.

2. **MPS-specific quirks**: The plan targets MPS + CUDA parity but doesn't mention known MPS issues (e.g., some operations falling back to CPU, different float precision defaults). Should at least log the device and note any expected divergence.

3. **Memory profiling**: V3 spec has `track_memory: bool = True`. The plan doesn't include memory logging. tracking.py has `log_gpu_memory()` — should be called periodically.

### Severity Summary

| ID | Severity | Issue |
|----|----------|-------|
| 1 | MEDIUM | Batch size choice undocumented, no OOM fallback guidance |
| 2 | LOW | Scheduler total_steps derivation not explicit |
| 3 | MEDIUM | Noise annealing not in presets; confounds no-lb ablation |
| 4 | LOW | noise_std/anneal_fraction not in CLI |
| 5 | LOW | format_version undefined |
| 6 | LOW | Step vs epoch equivalence undocumented |
| 7 | LOW | balance_tokens default ambiguous for training presets |
| 8 | MEDIUM | best-model.pt selection criteria undefined |
| 9 | LOW | Text-level eval split not mentioned in data flow |
| 10 | — | Non-issue, confirmed correct |
| 11 | LOW | max-example-chars not exposed |

**Overall Assessment: The plan is solid — 3 MEDIUM issues, no blockers.** The most consequential issue is #3 (noise in no-lb ablation) because it could confound a key experimental result. Issues #1 and #8 are practical gaps that should be resolved before implementation but don't change the architecture.

### Questions for GPT 5.3 Codex

1. **Noise in no-lb ablation**: Did you consider whether router noise should be disabled for the no-lb preset? If we're measuring collapse without load balancing, noise provides exploration that delays collapse — this is a confound. Should `noise_std=0.0` for no-lb?

2. **Batch size rationale**: What memory estimate led to `batch_size=2`? Is this based on profiling GPT-2 + MoE on MPS, or a conservative guess?

3. **best-model.pt**: What metric determines "best"? Lowest aggregate eval loss, or something per-domain?

4. **Scheduler coupling**: Is `num_training_steps` for cosine warmup always equal to `--max-steps`? What happens on resume — does the scheduler state restore correctly mid-schedule?

5. **RNG seeding**: The checkpoint includes `python_random_state`, `torch_rng_state`, `cuda_rng_state_if_available`. Does this also include numpy RNG? And is there a central `seed_everything()` at startup?

---

## GPT 5.3 Codex — Analysis 1

### Understanding the Current State

Claude Opus 4.6's review is high-quality and mostly well-calibrated. The core architecture in the Phase 4 plan is sound, and the flagged issues are mostly specification clarity and experimental hygiene rather than design blockers.

### Severity Recalibration

#### Agree (and keep priority high)

1. **Noise confound in `no-lb` ablation (Issue 3)** — keep as **MEDIUM-HIGH**.
   - If router noise remains active while LB is disabled, collapse timing becomes a mixed effect of "no LB" + "exploration noise."
   - For causal interpretation, `no-lb` should run with `noise_std=0.0` and no annealing.

2. **`best-model.pt` selection undefined (Issue 8)** — keep as **MEDIUM**.
   - This is operationally important for downstream inference and report consistency.
   - Must define one metric and one update rule.

3. **Batch-size fallback policy missing (Issue 1)** — keep as **MEDIUM**.
   - `2x4` is a good default, but no explicit OOM degradation ladder is documented for MPS.

#### Agree but lower practical urgency

4. **Scheduler coupling explicitness (Issue 2)** — **LOW**.
   - Simple one-line contract fix: `num_training_steps = max_steps`.

5. **`format_version` undefined (Issue 5)** — **LOW**.
   - Set `format_version=1`; fail fast on mismatch during resume.

6. **Step-vs-epoch equivalence note (Issue 6)** — **LOW**.
   - Helpful for interpretability, not a blocker.

7. **Preset clarity for `balance_tokens` (Issue 7)** — **LOW**.
   - Should be explicit for all non-shakedown presets.

8. **Data split detail in flow section (Issue 9)** — **LOW**.
   - Good documentation fix, code path already handles this.

9. **Expose `max-example-chars` (Issue 11)** — **LOW**.
   - Nice-to-have unless debugging pathological samples.

### Additional Gap Not Strongly Emphasized in Opus Review

**Reproducibility contract completeness** should be explicitly specified:
- Startup seeding for `random`, `torch`, `numpy`, and `torch.cuda.manual_seed_all` when CUDA exists.
- Checkpoint save/restore should include numpy RNG state if numpy is used in the training path.
- Resume should restore scheduler/optimizer/model and RNG deterministically before further steps.

This is **MEDIUM** for report credibility, though not architecture-blocking.

### Direct Answers to Opus Questions

1. **Should `no-lb` disable noise?**
   - **Yes.** Set `noise_std=0.0` and disable annealing for `no-lb` to avoid confounding collapse analysis.

2. **Why `batch_size=2`?**
   - It's a conservative cross-device baseline for GPT-2 + MoE on MPS/CUDA with reduced OOM risk.
   - Add explicit fallback: `2x4 -> 1x8 -> smaller block size`.

3. **What defines `best-model.pt`?**
   - Use **lowest aggregate eval loss** (`eval/loss`) over held-out eval set.
   - Update only at eval checkpoints; keep as model-only snapshot.

4. **Scheduler coupling and resume behavior?**
   - Set scheduler `num_training_steps = max_steps`.
   - On resume, restore scheduler state dict directly; do not recompute progress heuristically.

5. **RNG seeding completeness?**
   - Include `random`, `torch`, `numpy`, and CUDA RNG (if available), with centralized `seed_everything(seed)` and checkpointed RNG states for reproducible resume.

### Bottom Line

No architectural reset is needed. Resolve the three MEDIUM issues (noise confound, best-model rule, OOM fallback), tighten reproducibility/scheduler contracts, then proceed to implement `train.py` immediately.

---

## Claude Opus 4.6 — Response 1

### Assessment

Full agreement. GPT 5.3 Codex accepted all 11 findings, kept the same 3 at MEDIUM (with Issue 3 bumped to MEDIUM-HIGH — accepted), and added one new MEDIUM (reproducibility contract) that I flagged but didn't emphasize strongly enough. No pushback on any severity rating.

### Specific Responses

**Issue 3 → MEDIUM-HIGH (noise confound):** Accepted. GPT's answer is clean: `noise_std=0.0` and no annealing for `no-lb`. This means the `no-lb` preset definition should include `noise_std=0.0` explicitly (not just `lb=0.0`). Implementation note: `install_moe_layers()` takes `noise_std` as a parameter, so this is a one-line preset change.

**Issue 8 → MEDIUM (best-model.pt):** Accepted. Lowest aggregate `eval/loss`, updated at eval checkpoints, saved as model-only snapshot. Clear and implementable.

**Issue 1 → MEDIUM (OOM fallback):** Accepted. GPT's fallback ladder (`2×4 → 1×8 → smaller block_size`) is practical. This is documentation, not code — add to the plan as a troubleshooting note.

**Reproducibility (new MEDIUM):** Accepted. This was in my "What's Not in the Plan" section but I didn't assign it a severity. MEDIUM is right — report credibility depends on stating exactly how reproducibility is handled. The `seed_everything(seed)` utility and numpy RNG in checkpoints are both low-effort, high-value additions.

### Remaining LOW items

All LOW items (Issues 2, 4, 5, 6, 7, 9, 11) can be addressed inline during implementation. None require plan revision.

### Verdict

**Converged. No open disagreements.** We agree on the plan architecture, all severity ratings, and all resolutions. The plan needs 4 amendments before implementation begins:

1. `no-lb` preset: add `noise_std=0.0`
2. `best-model.pt`: lowest aggregate eval loss, model-only snapshot
3. Add OOM fallback ladder to plan
4. Add `seed_everything()` + numpy RNG state to reproducibility contract

---

## Convergence Summary

### Agreed Points (Both Models)

1. **Plan architecture is sound** — single-script dual-mode, step-based loop, preset system, dual checkpoints, shakedown gate, 3-consecutive-checks collapse criterion are all endorsed.
2. **Step-based > epoch-based** for this project's packed-dataset design.
3. **Dual checkpoint format** (full-resume + model-only) is a practical improvement over V3.
4. **All 11 originally flagged issues are valid** with agreed severities.

### Resolved Issues (4 MEDIUM, 7 LOW)

| ID | Final Severity | Resolution |
|----|---------------|------------|
| 3 | **MEDIUM-HIGH** | `no-lb` preset must set `noise_std=0.0` and disable annealing to avoid confounding collapse measurement |
| 8 | **MEDIUM** | `best-model.pt` = lowest aggregate `eval/loss`, updated at eval checkpoints, saved as model-only snapshot |
| 1 | **MEDIUM** | Document OOM fallback ladder: `2×4 → 1×8 → reduce block_size`. Add expected peak memory note |
| R | **MEDIUM** | Add `seed_everything(seed)` at startup (random, torch, numpy, CUDA). Include numpy RNG state in checkpoints |
| 2 | LOW | Scheduler `num_training_steps = max_steps` (one-line contract) |
| 4 | LOW | `noise_std` and `anneal_fraction` not in CLI — accept defaults, document in presets |
| 5 | LOW | `format_version=1`, fail fast on mismatch |
| 6 | LOW | Document epoch-equivalent per preset for interpretability |
| 7 | LOW | Explicit `balance_tokens=true` in all non-shakedown presets |
| 9 | LOW | Mention `split_texts_for_eval()` in data flow section |
| 11 | LOW | `max-example-chars` not exposed — acceptable |

### Final Amended Preset Definitions

```
shakedown:  max_steps=100,  size_mb=1,   balance_tokens=false, no MoE or dense-only
dense:      max_steps=5000, size_mb=10,  balance_tokens=true,  no MoE, lb=0, z=0
moe-main:   max_steps=10000, size_mb=10, balance_tokens=true,  top1, lb=0.01, z=0.001, noise_std=0.1
no-lb:      max_steps=2000, size_mb=10,  balance_tokens=true,  top1, lb=0.0, z=0.001, noise_std=0.0, collapse_detect=true
top2:       max_steps=3000, size_mb=10,  balance_tokens=true,  top2, lb=0.01, z=0.001, noise_std=0.1
```

### Action Order

1. Amend `PHASE-4-TRAINING-PLAN.md` with the 4 resolved MEDIUM items
2. Implement `train.py` following amended plan
3. Run shakedown gate (dense + MoE)
4. Execute run schedule: dense → moe-main → no-lb → top2 (budget permitting)
