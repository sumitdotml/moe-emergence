# MoE Emergence - Agent Instructions

This file provides context and workflows for AI coding assistants working on this project.

---

# 0. Documentation Workflow (IMPORTANT)

This project maintains rigorous documentation for a technical report. **Every significant action should be documented.**

## Skills

Documentation skills live in `.claude/skills/`. Claude will auto-invoke some of these when relevant (e.g., when you make a design choice), or you can invoke them directly with `/skill-name`.

| Skill             | Purpose                     | Example                                         | Auto-invoke |
| ----------------- | --------------------------- | ----------------------------------------------- | ----------- |
| `/doc-decision`   | Log a design choice         | `/doc-decision chose top-1 routing over top-2`  | Yes         |
| `/doc-experiment` | Log a training run          | `/doc-experiment completed main MoE run`        | Yes         |
| `/doc-fix`        | Document a bug fix          | `/doc-fix fixed gradient flow in top-1 routing` | Yes         |
| `/doc-paper`      | Add literature notes        | `/doc-paper Switch Transformer Fedus 2021`      | Yes         |
| `/doc-review`     | Code review against V3 spec | `/doc-review moe_emergence/moe.py`              | No          |
| `/doc-status`     | Project status summary      | `/doc-status`                                   | No          |

## When to Document

| Event                                | Action            | Priority |
| ------------------------------------ | ----------------- | -------- |
| Made a design choice between options | `/doc-decision`   | HIGH     |
| Completed a training run             | `/doc-experiment` | HIGH     |
| Fixed a bug                          | `/doc-fix`        | HIGH     |
| Reviewed code before training        | `/doc-review`     | HIGH     |
| Read a paper you'll cite             | `/doc-paper`      | MEDIUM   |
| Starting a new session               | `/doc-status`     | MEDIUM   |

## Directory Structure

```
.claude/
├── settings.local.json
└── skills/                   # Documentation skills
    ├── doc-decision/
    ├── doc-experiment/
    ├── doc-fix/
    ├── doc-paper/
    ├── doc-review/
    └── doc-status/

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

- `moe_emergence/`: Core Python modules — `moe.py`, `ffn.py`, `gpt2_moe.py`, `data.py`, `tracking.py`, `train.py`, `gpt2_inference.py`.
- `notebooks/`: Exploratory experiments and visualizations.
- `docs/project-design/`: Design notes and project plans.
- `README.md`: Goals, current status, and references.

## Build, Test, and Development Commands

- `uv sync`: Install dependencies from `uv.lock` (recommended local setup).
- `uv run python -m moe_emergence.train --preset shakedown --run-name test`: Run a shakedown training test.
- `uv run python moe_emergence/verify_gpt2_integration.py`: Run Phase 2 verification (10 comprehensive tests).
- `uv run python moe_emergence/gpt2_inference.py --moe`: Interactive inference playground with routing stats.
- `uv run ruff format moe_emergence/`: Auto-format code with Ruff.
- `uv run ruff check moe_emergence/`: Lint for style and correctness issues.

## Coding Style & Naming Conventions

- Indentation: 4 spaces; line length: 88 (Ruff defaults in `pyproject.toml`).
- Use double quotes in Python where formatting applies (Ruff formatter).
- Naming: `snake_case` for functions and modules, `PascalCase` for classes, `ALL_CAPS` for constants.
- Add type hints for new public APIs and non-trivial functions.

## Testing Guidelines

- There is no formal test suite yet. Use `python moe_emergence/training_demo.py` as the current sanity check.
- If you add tests, place them under `tests/` and name files `test_*.py` (consider pytest if introducing automated tests).

## Commit & Pull Request Guidelines

- Commit history uses short, lowercase, descriptive messages without prefixes. Example: `add moe routing demo`.
- PRs should include a brief summary, motivation, and tests run. Link issues if relevant and attach plots or screenshots for routing/visualization changes.

## Data, Outputs, and Artifacts

- Keep large datasets, model checkpoints, and generated plots out of version control unless explicitly requested.
- When adding notebooks, clear or minimize outputs to keep diffs small and focused.
- **All cached data must live in `.cache/` at the repository root** — not in `~/.cache/` or other system locations. This includes HuggingFace datasets, model weights, and any downloaded files. The `.cache/` directory is gitignored.
- `checkpoints/` and `wandb/` are local-run artifacts and should remain out of git; clean old runs after validation to avoid disk bloat.

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

Code review issues (all fixed):

- `moe.py` fixes in commit `eea9294` - see `docs/code-reviews/001-2025-12-23-moe-py-review.md`
- `gpt2_moe.py` fixes in commit `ffc77ab` - see `docs/code-reviews/003-2025-12-23-gpt2-moe-fix.md`
- Loss dedup + test hardening in commit `c929d8c` - see `docs/code-reviews/004-2025-12-23-loss-dedup-and-tests.md`

Open items from cross-model audit (debate 008):

- ~~**P1: Import architecture** — Fixed in `3541faa`. All imports now use `moe_emergence.*` package form.~~
- **P1: Stale docs** — `docs/DATA-PIPELINE.md` and V3 spec contain outdated dataset refs. DATA-PIPELINE.md marked superseded; V3 snippets are historical.
- ~~**High P2: Eval split formula** — Fixed in `39b069e`. Docstring now accurately describes small-n behavior.~~

## Current Status (2026-02-24)

**Completed:**

- Phase 1 (MoE components) [DONE]
- Phase 2 (GPT-2 integration) [DONE]
  - Full integration verification: 10/10 tests passed
  - See: `docs/experiments/run-001-gpt2-integration-verification.md`
  - Commit: `a15683e`
- Phase 3 (Dataset preparation) [DONE]
  - Sequence packing implemented
  - `PackedMixedDomainDataset` with token balancing
  - W&B tracking utilities (`moe_emergence/tracking.py`)
  - Multi-model debates: data pipeline (005\*.md), tracking review (006)
  - Full project audit (007) + cross-model convergence review (008)
  - Phase 4 training plan reviewed and amended (debate 009)
- Phase 4 (Training) [DONE — all 4 budgeted runs complete]
  - Infrastructure: `train.py` with presets, grad accum, eval, checkpointing/resume, collapse detection
  - GPU setup: PrimeIntellect RTX 4090, see `docs/GPU-SETUP.md`
  - **run-003**: GPU shakedown — both dense & MoE passed (`docs/experiments/run-003-gpu-shakedown-primeintellect.md`)
  - **run-004**: Dense baseline — eval/loss=2.157, 5000 steps, ~30min (`docs/experiments/run-004-dense-baseline.md`)
  - **run-005**: MoE main run — eval/loss=2.080, 10000 steps, ~85min (`docs/experiments/run-005-moe-main.md`)
  - **run-006**: No-LB ablation — collapsed at step 500, confirms LB loss is essential (`docs/experiments/run-006-no-lb-ablation.md`)
  - **run-007**: Top-2 directional — eval/loss=2.077, 10000 steps, ~48min, marginal 0.14% over top-1 (`docs/experiments/run-007-top2-directional.md`)
  - MoE beats dense by 3.6% overall, 14% on math, 2.1% on code; dense wins prose by 1.6%
  - Top-2 does not meaningfully outperform top-1; validates top-1 routing choice
  - Without LB loss, expert collapse within 500 steps; z-loss alone doesn't prevent it

**Current Phase:** Phase 5 — post-training analysis and visualization

**Training Plan:** `docs/project-design/PHASE-4-TRAINING-PLAN.md` (reviewed, amended with debate 009 resolutions)

**Phase 4 Implementation (commit `5efdfae`):**
- `train.py`: presets, gradient accumulation, eval loop, checkpointing/resume, collapse detection
- Model-only snapshots: `.safetensors` + `.json` sidecar (clone for GPT-2 tied weights)
- Full resume: `.pt` with mode + architecture validation
- `gpt2_inference.py`: updated to load safetensors, mode-aware (dense vs MoE)

**Phase 4 Hardening (2026-02-18):**
- Per-domain eval metrics fixed to sequence-level CE grouped by domain
- W&B eval perplexity now uses LM-based eval perplexity from `run_eval()` (no recompute from aggregate loss)
- Empty-dataset fail-fast guards added for train/eval block construction
- CUDA-safe input pipeline defaults added (`num_workers=2`, `pin_memory=True`, `persistent_workers=True`, non-blocking transfer)
- Per-domain training metrics added (`train/loss_code`, `train/loss_math`, `train/loss_prose`)
- Resume RNG restoration hardened for MPS/CPU byte-state requirements
- Documentation added: `docs/code-reviews/006-2026-02-18-phase4-training-review.md`, `docs/code-reviews/007-2026-02-18-phase4-training-fix.md`, `docs/experiments/run-002-phase4-hardening-shakedown.md`

**Phase 4 Training Results (2026-02-23/24/25):**
- GPU: PrimeIntellect RTX 4090 24GB, $0.61/hr (Norway for runs 003-005, Romania for 006-007)
- Shakedown: both modes passed, lb_loss=1.05, no collapse (run-003)
- Dense baseline: eval/loss=2.157, ppl=8.64, ~25.7k tok/s, ~30min (run-004)
- MoE main: eval/loss=2.080, ppl=7.91, ~14.2k tok/s, ~85min (run-005)
- No-LB ablation: collapsed at step 500, expert 1 in layer 9 captured 73.6% of tokens (run-006)
- Top-2 directional: eval/loss=2.077, ppl=7.89, ~18.9k tok/s, ~48min (run-007)
- MoE crossed dense at step ~3600 (36% of training), plateaued by step ~8000
- Total GPU cost: ~$2.79 (includes setup/idle time, shakedown, all 4 runs, disk)
- All artifacts downloaded locally to `checkpoints/` and uploaded to HuggingFace

**Verified Decisions:**

| Decision            | Choice                                   | Notes                                                     |
| ------------------- | ---------------------------------------- | --------------------------------------------------------- |
| Math dataset        | MathQA (allenai)                         | 29K examples, ~11.3MB, Apache 2.0, loaded from source ZIP |
| MathQA formatting   | `{Problem}\n\n{Rationale}` (no prefixes) | See decision 007 (revised)                                |
| Code dataset        | CodeParrot-clean                         | Diverse Python, multiple licenses, see decision 010       |
| Prose dataset       | AllenAI C4 (en)                          | Natural web text, well-filtered, see decision 010         |
| Token balancing     | `--balance-tokens` required for training | 1.9x imbalance without it, see decision 005 empirical     |
| Shuffle truncation  | Shuffle blocks before truncating         | Defensive measure, see decision 012                       |
| Experiment tracking | W&B                                      | See decision 009, `moe_emergence/tracking.py`             |
| No-lb noise         | `noise_std=0.0`, no annealing            | Avoids confounding collapse measurement, see debate 009   |
| Best model metric   | Lowest aggregate eval loss               | Model-only snapshot, updated at eval checkpoints          |
| Checkpoint format   | `.pt` resume + `.safetensors` snapshots  | Avoids pickle for model-only artifacts, see training plan |

**Pending Investigation:**

These items require verification before implementation. Must not assume they are correct.

| Item                     | What Needs Investigation                                          | Status                                  |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------- |
| Train/eval split formula | Is `max(20, int(n * 0.05))` the right approach? Verify rationale. | **DONE** — uses Decision 008 formula    |
| Code dataset             | CodeParrot-clean vs StarCoderData — verify samples                | **DONE** — CodeParrot-clean             |
| Prose dataset            | WikiText-103 vs OpenWebText vs C4 vs FineWeb — verify samples     | **DONE** — AllenAI C4 (en)              |
| Formatting artifacts     | Check for whitespace/invisible char anomalies in all datasets     | **DONE** — see decision 011             |
| Shuffle buffer formula   | Is `max(1000, size_mb*200)` justified? Where did this come from?  | **DONE** — not needed, see decision 006 |

**Next Actions:**

1. ~~Run budgeted experiments: dense → moe-main~~ [DONE]
2. ~~Run ablation experiments: no-lb → top-2~~ [DONE]
3. ~~Upload models to HuggingFace~~ [DONE — `sumitdotml/moe-emergence`]
4. Phase 5: post-training analysis and visualization (all runs on CPU/MPS locally)
   - Domain-level expert activation analysis (which experts specialize on which domains)
   - Fine-grained token-type routing (Python keywords vs operators vs strings)
   - Router entropy over training visualization
   - Collapse comparison visualization (with LB vs without LB)
   - Publication-quality figures for technical report

## Budget Constraint

$80 GPU budget. Total spent: ~$2.79 (includes setup/idle/disk overhead). Remaining: ~$77.21.

All training runs complete:
1. ~~Dense baseline~~ ($0.31)
2. ~~MoE main run~~ ($0.86)
3. ~~No-LB ablation~~ ($0.05)
4. ~~Top-2 directional~~ ($0.49)
5. Setup/idle/disk overhead (~$1.08)
