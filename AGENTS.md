# MoE Emergence - Agent Instructions

This file provides context and workflows for AI coding assistants working on this project.

---

# 0. Documentation Workflow (IMPORTANT)

This project maintains rigorous documentation for a technical report. **Every significant action should be documented.**

## Quick Commands

| Command           | Purpose                     | Example                                         |
| ----------------- | --------------------------- | ----------------------------------------------- |
| `/doc-decision`   | Log a design choice         | `/doc-decision chose top-1 routing over top-2`  |
| `/doc-experiment` | Log a training run          | `/doc-experiment completed main MoE run`        |
| `/doc-review`     | Code review against V3 spec | `/doc-review moe-emergence/moe.py`              |
| `/doc-paper`      | Add literature notes        | `/doc-paper Switch Transformer Fedus 2021`      |
| `/doc-fix`        | Document a bug fix          | `/doc-fix fixed gradient flow in top-1 routing` |
| `/doc-status`     | Project status summary      | `/doc-status`                                   |

## When to Document

| Event                                | Action            | Priority |
| ------------------------------------ | ----------------- | -------- |
| Made a design choice between options | `/doc-decision`   | HIGH     |
| Completed a training run             | `/doc-experiment` | HIGH     |
| Fixed a bug                          | `/doc-fix`        | HIGH     |
| Reviewed code before training        | `/doc-review`     | HIGH     |
| Read a paper you'll cite             | `/doc-paper`      | MEDIUM   |
| Starting a new session               | `/doc-status`     | MEDIUM   |

## Documentation Structure

```
docs/
├── README.md                 # Index
├── code-reviews/             # Code reviews with commit hashes
├── decisions/                # Why choices were made
├── experiments/              # Training run logs
├── models-debate/            # Model discussion logs and critiques
└── literature/               # Paper notes
```

## Key Principle

Always capture the **commit hash** when documenting. This creates an audit trail linking documentation to exact code state.

---

# 1. Remove AI code slop

Check the diff against main, and remove all AI generated slop introduced in this branch.

This includes:

- Extra comments that a human wouldn't add or is inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted / validated codepaths)
- Casts to any to get around type issues
- Variables that are only created to be immediately returned on the next line -- inline into the return directly
- Any other style that is inconsistent with the file

Report at the end with only a 1-3 sentence summary of what you changed

# 2. Repository Guidelines

## Project Structure & Module Organization

- `moe-emergence/`: Core Python modules for MoE routing and GPT-2 integration (`moe.py`, `ffn.py`, `gpt2_moe.py`, `training_demo.py`).
- `notebooks/`: Exploratory experiments and visualizations.
- `project-design/`: Design notes and project plans.
- `README.md`: Goals, current status, and references.

## Build, Test, and Development Commands

- `uv sync`: Install dependencies from `uv.lock` (recommended local setup).
- `uv run python moe-emergence/verify_gpt2_integration.py`: Run Phase 2 verification (10 comprehensive tests).
- `uv run python moe-emergence/gpt2_inference.py --moe`: Interactive inference playground with routing stats.
- `python moe-emergence/training_demo.py`: Run a small MoE training demo as a smoke test.
- `ruff format moe-emergence/`: Auto-format code with Ruff.
- `ruff check moe-emergence/`: Lint for style and correctness issues.

## Coding Style & Naming Conventions

- Indentation: 4 spaces; line length: 88 (Ruff defaults in `pyproject.toml`).
- Use double quotes in Python where formatting applies (Ruff formatter).
- Naming: `snake_case` for functions and modules, `PascalCase` for classes, `ALL_CAPS` for constants.
- Add type hints for new public APIs and non-trivial functions.

## Testing Guidelines

- There is no formal test suite yet. Use `python moe-emergence/training_demo.py` as the current sanity check.
- If you add tests, place them under `tests/` and name files `test_*.py` (consider pytest if introducing automated tests).

## Commit & Pull Request Guidelines

- Commit history uses short, lowercase, descriptive messages without prefixes. Example: `add moe routing demo`.
- PRs should include a brief summary, motivation, and tests run. Link issues if relevant and attach plots or screenshots for routing/visualization changes.

## Data, Outputs, and Artifacts

- Keep large datasets, model checkpoints, and generated plots out of version control unless explicitly requested.
- When adding notebooks, clear or minimize outputs to keep diffs small and focused.

---

# 3. Project-Specific Context

## What This Project Is

Training a small MoE model on 3 domains (code, math, prose) to demonstrate expert specialization emergence. Goal is visualizations and a technical report, not SOTA performance.

## Key Design Document

**Read `docs/project-design/MOE-PROJECT-DESIGN-V3.md` before making implementation changes.** This is the authoritative spec created through multi-model review.

## Architecture Decisions (Already Made)

| Decision    | Choice                       | Rationale                                     |
| ----------- | ---------------------------- | --------------------------------------------- |
| Base model  | GPT-2 small                  | Controlled baseline, well-documented behavior |
| MoE layers  | Last 4 (8-11)                | Later layers show domain-specific features    |
| Experts     | 8, top-1 routing             | Matches Switch Transformer, budget-friendly   |
| Expert init | deepcopy of MLP + tiny noise | Warm-start preserves pretrained knowledge     |

## Critical Implementation Details

1. **STE for top-1**: Forward uses weight=1.0, backward flows through soft probability
2. **Load balancing loss**: `n_experts * Σ(f_i * P_i)`, minimum is 1.0 at perfect balance
3. **Z-loss**: `mean(logsumexp(logits)²)`, stabilizes router
4. **Warm-start**: Use `copy.deepcopy(original_mlp)`, NOT manual architecture recreation

## Known Issues

All issues from code reviews have been fixed:

- `moe.py` fixes in commit `eea9294` - see `docs/code-reviews/001-2025-12-23-moe-py-review.md`
- `gpt2_moe.py` fixes in commit `ffc77ab` - see `docs/code-reviews/003-2025-12-23-gpt2-moe-fix.md`
- Loss dedup + test hardening in commit `c929d8c` - see `docs/code-reviews/004-2025-12-23-loss-dedup-and-tests.md`

## Current Status (2025-01-21)

**Completed:**

- Phase 1 (MoE components) [DONE]
- Phase 2 (GPT-2 integration) [DONE]
  - Full integration verification: 10/10 tests passed
  - See: `docs/experiments/run-001-gpt2-integration-verification.md`
  - Commit: `a15683e`
- Phase 3 (Dataset preparation) [IN PROGRESS]
  - Sequence packing implemented
  - `PackedMixedDomainDataset` implemented
  - Multi-model debate completed (11 passes, see `docs/models-debate/005*.md`)
  - **5 critical data hygiene issues identified and fixes locked**

**Current Phase:** Phase 3 (Dataset Preparation) — Data Hygiene Fixes

**Blockers (No Training Until Resolved):**

The multi-model debate (Opus 4.5 + GPT-5.2 + Gemini 3) identified 5 experiment-invalidating issues. These are now locked decisions:

| Issue                     | Fix                                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| CodeParrot streaming bias | `ds.shuffle(buffer_size=max(1000, size_mb*200), seed=seed)`                                                             |
| Math prefix leakage       | Randomize both from `["Problem:", "Question:", "Given:"]` + `["Solution:", "Answer:", "Therefore:"]` (no empty strings) |
| Train/eval leakage        | Split at TEXT level BEFORE packing                                                                                      |
| No validation split       | `min(max(10, int(n_texts*0.05)), int(n_texts*0.10))` texts as holdout                                                   |
| Subdomain provenance loss | Add `subdomain` field: `"gsm8k"` or `"competition_math"`                                                                |

**Next Action:**

1. Create `/doc-decision` entries for the 4 locked decisions (streaming, prefixes, validation, subdomain)
2. Update `moe-emergence/data.py`:
   - Add shuffle buffer with seed for CodeParrot
   - Randomize both math prefixes (problem AND solution)
   - Split texts BEFORE packing (not blocks after)
   - Pack train/eval separately per domain
   - Add `subdomain` field for math
   - Log buffer_size, holdout text/block counts, seed
3. Update `docs/DATA-PIPELINE.md`:
   - Document text-level split strategy
   - Clarify `--size-mb` is total (train+eval pre-split)
   - Note WikiText-103 = "encyclopedic prose" limitation
4. Run verification: `uv run python moe-emergence/data.py --size-mb 10 --block-size 512`
5. Confirm no train/eval leakage, record counts in `docs/DATA-PIPELINE.md`

## Budget Constraint

$80 GPU budget. Prioritize:

1. Dense baseline (required)
2. MoE main run (required)
3. No-LB collapse ablation (required, early-stop)
4. Top-2 ablation (optional, short run)
