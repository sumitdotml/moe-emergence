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

**Read `project-design/MOE-PROJECT-DESIGN-V3.md` before making implementation changes.** This is the authoritative spec created through multi-model review.

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

All issues from code review 2025-12-23 have been fixed:

- `moe.py` fixes in commit `eea9294` - see `docs/code-reviews/001-2025-12-23-moe-py-review.md`
- `gpt2_moe.py` fixes in commit `1f2b581` - see `docs/code-reviews/003-2025-12-23-gpt2-moe-fix.md`

## Budget Constraint

$80 GPU budget. Prioritize:

1. Dense baseline (required)
2. MoE main run (required)
3. No-LB collapse ablation (required, early-stop)
4. Top-2 ablation (optional, short run)
