# Project Critical Analysis Book

**Date:** 2026-02-07  
**Audit Type:** End-to-end technical and documentation audit  
**Audited Commit:** `8d49faae2b6579a3754897a9be780c28b59e482d`  
**Branch:** `main`  
**Scope:** AGENTS instructions, design/spec docs, decisions, code reviews, experiments, core code, and runnable checks

---

## Executive Position

The project is **conceptually strong and unusually well-documented** for its scale, but it is **not yet in execution-ready "critical phase" condition** because Phase 4 training infrastructure is still missing and there are several coherence issues between "authoritative" docs and current code reality.

Most important: **the core MoE/GPT-2 integration is working and verified**, but training/evaluation orchestration and final scientific execution guardrails are not yet operational.

---

## Master Track List (Checklist)

This is the recommended one-by-one checklist for future critical sessions.

| # | Track Item | Why it matters | Status in this audit | Primary Evidence |
|---|------------|----------------|----------------------|------------------|
| 1 | Governance baseline (`AGENTS.md`) | Defines constraints and source of truth | DONE | `AGENTS.md` |
| 2 | Phase and milestone truth | Prevents planning off stale assumptions | DONE | `README.md`, `docs/README.md`, git log |
| 3 | Design-spec alignment (V3 vs implementation) | Avoids building to contradictory requirements | DONE | `docs/project-design/MOE-PROJECT-DESIGN-V3.md`, code |
| 4 | Decision-log coherence | Ensures rationale and code still match | DONE | `docs/decisions/001-012*.md` |
| 5 | Code-review closure verification | Confirms known bugs actually resolved | DONE | `docs/code-reviews/*.md`, runtime checks |
| 6 | Data pipeline validity + leakage controls | Protects experiment credibility | DONE | `moe_emergence/data.py`, decisions 006/008/010/012 |
| 7 | Model core correctness (router, STE, aux losses) | Central to MoE behavior claims | DONE | `moe_emergence/moe.py`, `moe_emergence/gpt2_moe.py` |
| 8 | Experiment tracking readiness | Needed for report-grade outputs | DONE | `moe_emergence/tracking.py`, debate 006 |
| 9 | Reproducibility + packaging integrity | Needed for repeatability and handoff quality | DONE | `pyproject.toml`, import tests |
| 10 | Runtime sanity checks | Guards against "docs say yes, code says no" | DONE | `ruff`, `gpt2_moe.py`, `training_demo.py`, `verify_gpt2_integration.py` |
| 11 | Budget feasibility against required runs | Ensures critical phase can actually finish | DONE | AGENTS budget + V3 run plan |
| 12 | Debate agenda preparation | Enables high-value technical review round | DONE | Section "Debate Agenda" below |

---

## Where The Project Actually Is

### Confirmed state
- Phase 1 done (MoE primitives).
- Phase 2 done and re-verified (10/10 integration checks pass).
- Phase 3 done (dataset pipeline, split-before-pack, final dataset choices, tracking utility).
- **Phase 4 not done** (training loop, checkpointing, dense baseline run still pending).

### Runtime verification run in this audit
- `uv run ruff check moe_emergence` -> passed.
- `uv run python moe_emergence/gpt2_moe.py` -> smoke test passed.
- `uv run python moe_emergence/training_demo.py` -> passed.
- `uv run python moe_emergence/verify_gpt2_integration.py` -> full 10/10 tests passed again at current commit.

---

## Critical Findings (Severity Ordered)

## P0 - Likely to block or materially damage the critical phase

### 1) Phase 4 core execution path is still missing
There is no complete training entrypoint implementing the required LM + LB + Z loop, eval, checkpointing, and experiment runs.

- Evidence:
  - `AGENTS.md` shows Phase 4 as current and pending.
  - `README.md` still lists training loop/checkpointing/baseline as unchecked.
  - `README.md` references `train.py` as not written.

**Impact:** You cannot yet run the required dense baseline + MoE main + no-LB ablation sequence that the report depends on.

---

### 2) Package import path architecture is fragile and fails package-style usage
Several modules rely on script-local imports (`from moe import ...`, `from gpt2_moe import ...`, `from ffn import ...`).

- Evidence:
  - `moe_emergence/gpt2_moe.py` line 22
  - `moe_emergence/moe.py` line 28
  - `moe_emergence/verify_gpt2_integration.py` line 29
  - `moe_emergence/training_demo.py` line 16
  - `moe_emergence/gpt2_inference.py` lines 119, 140, 195
  - `uv run python -c "import moe_emergence.gpt2_moe"` fails with `ModuleNotFoundError: No module named 'moe'`

**Impact:** Works when run as file paths, fails in package/module import contexts; this is a handoff/reusability risk and can break CI/module-mode execution.

---

### 3) "Authoritative spec" drift: key docs are stale/contradictory
The project claims V3 is authoritative, but V3 and `docs/DATA-PIPELINE.md` contain stale dataset and method details.

- Evidence:
  - `docs/DATA-PIPELINE.md` still states no train/eval split and old datasets (GSM8K/MATH/WikiText).
  - `docs/project-design/MOE-PROJECT-DESIGN-V3.md` still includes old code snippets using GSM8K/OpenWebText.
  - V3 includes at least one stale top-1 statement inconsistent with final STE formulation.

**Impact:** High risk that future implementation decisions follow stale instructions and reintroduce resolved issues.

---

## P1 - Serious risks (not immediate project death, but high leverage)

### 4) Eval split formula semantics are internally inconsistent
`compute_eval_count()` uses:
`min(max(10, int(n * 0.05)), int(n * 0.10))`
with doc text claiming "at least 10 texts."

- Evidence:
  - `moe_emergence/data.py` lines 37-48
  - `docs/decisions/008-text-level-validation-split.md` lines 73-81

This formula does **not** guarantee >=10 for small `n` (for `n<100`, it can be <10; for very small `n`, even 0).

**Impact:** Can silently produce tiny/empty eval sets in low-data debug runs; confuses interpretation and contradicts stated rationale.

---

### 5) Cache policy mismatch in prose loader
Project policy says caches must live under repo `.cache/`. Code and tokenizer loaders follow this in some places, but prose loader does not pass `cache_dir`.

- Evidence:
  - Policy in `AGENTS.md` (cache rule section).
  - `moe_emergence/data.py` lines 316-321 (`load_dataset("allenai/c4", ...)` without `cache_dir`).

**Impact:** Hidden cache sprawl outside repo; weaker reproducibility and policy non-compliance.

---

### 6) Leakage assertion strategy can false-fail on exact duplicate texts
Leakage check uses set intersection of text strings between train/eval.

- Evidence:
  - `moe_emergence/data.py` lines 589-591

This is strict and catches true leakage, but exact duplicates in source data can also trigger failure even without split logic bugs.

**Impact:** Potential brittle failures on larger/noisier data pulls.

---

## P2 - Caveats and technical debt (mostly acceptable at this scale)

### 7) Sequential expert dispatch is intentionally non-optimized
Expert dispatch is Python-loop based in both standalone MoE and GPT-2 wrapper.

**Impact:** Throughput overhead, but acceptable for this project scale and budget objective.

---

### 8) Tracking module is implemented, but full training-loop integration is still missing
`tracking.py` looks coherent post-review, but the actual training loop that should feed it is not yet in place.

**Impact:** Good instrumentation exists, but the experiment pipeline is incomplete until Phase 4 is implemented.

---

### 9) Some files still contain explicit "generated/unverified" provenance and tutorial-style verbosity
Examples:
- `moe_emergence/training_demo.py`
- `moe_emergence/verify_formatting.py`

**Impact:** Not a functional blocker, but weakens production-readability and can dilute confidence during external review.

---

## Strengths (What Was Done Very Well)

1. **Review discipline is excellent for project size**  
   Multi-pass model debates + explicit fix reports + decision logs with commit links.

2. **Core MoE correctness is strong**  
   Router outputs separate clean/noisy probabilities; STE behavior and aux losses are implemented and empirically revalidated.

3. **Methodological awareness is high**  
   The team proactively addressed train/eval leakage, token balance confounds, and entropy semantics.

4. **Budget realism is explicit**  
   Required-vs-optional runs are clearly prioritized for an $80 cap.

5. **Audit trail quality is unusually good**  
   Chronological commit history aligns with decision progression and fixes.

---

## "Fine for Small Scale" vs "Will Break"

### Fine for this project scale
- No capacity factor / token dropping.
- Non-fused expert dispatch.
- Not implementing stream shuffle after empirical investigation.
- Limited subdomain metadata granularity.

### Will likely break critical-phase credibility if left unresolved
- Missing Phase 4 training execution path.
- Spec/document source-of-truth drift.
- Package import fragility.
- Ambiguous eval split semantics.

---

## Debate Agenda (Recommended for the next critical session)

1. **Source-of-truth lock**
   - Decide one canonical implementation contract doc and mark others as historical.
   - Remove or clearly deprecate stale sections in `docs/DATA-PIPELINE.md` and V3 snippets.

2. **Phase 4 minimum shippable implementation**
   - Implement one executable training script with:
     - LM + LB + Z losses
     - periodic eval
     - checkpointing/resume
     - tracking hooks
     - collapse detector for no-LB run

3. **Import architecture cleanup**
   - Convert internal imports to package-safe form (`moe_emergence.*`) or explicit relative imports.
   - Validate both invocation styles: file path and `python -m`.

4. **Eval split policy clarification**
   - Keep current formula but fix wording, or replace formula to match intended guarantee.
   - Add unit tests for `compute_eval_count()` edge cases.

5. **Cache compliance**
   - Ensure all dataset/tokenizer loaders use repo-local cache directory.

6. **Run plan freeze**
   - Finalize step counts/checkpoint cadence for:
     - Dense baseline (required)
     - MoE main run (required)
     - No-LB collapse run (required)
     - top-2 (optional)

---

## Recommended Immediate Action Order

1. Fix import architecture (low effort, high leverage).
2. Resolve spec drift and mark canonical doc.
3. Implement Phase 4 training loop + checkpointing.
4. Run dense baseline first (required control).
5. Run MoE main and no-LB ablation with agreed logging schema.

---

## Final Assessment

The staff-engineering and decision-thinking work is **substantively strong**, especially around risk identification and auditability. The project is now at the exact transition point where documentation rigor must be converted into execution rigor: **Phase 4 implementation discipline is now the bottleneck, not conceptual design quality**.
