# Phase 4 Training Plan

> **Reviewed and amended** — Cross-model convergence review (Opus 4.6 + GPT 5.3 Codex) on 2026-02-08.
> See `docs/models-debate/009-PHASE4-PLAN-CONVERGENCE-2026-02-08.md` for the full debate log.

## Summary
This plan defines a decision-complete implementation for Phase 4 training infrastructure: `train.py`, eval loop, tracking integration, checkpointing/resume, mandatory shakedown runs, and run presets for dense baseline and MoE experiments.
It is optimized for dual-device parity (MPS + CUDA) and budget-constrained execution.

## Scope
1. Implement `moe_emergence/train.py`.
2. Support dense and MoE training paths in one script.
3. Integrate `moe_emergence/tracking.py`.
4. Add dual checkpoint outputs (full-resume `.pt` + model-only `.safetensors`).
5. Add mandatory pre-run shakedown gate.
6. Add run presets for dense, moe-main, no-lb, and top2.
7. Document run commands and acceptance criteria.

## Out of Scope
1. Import normalization to `moe_emergence.*`.
2. `compute_eval_count()` formula/docstring reconciliation.
3. Dispatch-kernel optimization.
4. Historical doc cleanup beyond already-added superseded banner.

## Public Interface Contract (`train.py`)
`moe_emergence/train.py` must expose this CLI:

| Arg | Type | Default | Notes |
|---|---|---|---|
| `--preset` | str | required | `shakedown`, `dense`, `moe-main`, `no-lb`, `top2` |
| `--run-name` | str | required | Run directory key |
| `--output-dir` | str | `checkpoints` | Relative to repo root |
| `--resume` | str | None | Full checkpoint path only |
| `--device` | str | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--seed` | int | `42` | Global seed |
| `--size-mb` | float | `10.0` | Per domain, pre-split |
| `--block-size` | int | `512` | Packed block size |
| `--balance-tokens` | flag | preset | True for training runs |
| `--batch-size` | int | `2` | Micro-batch size |
| `--grad-accum-steps` | int | `4` | Effective batch control |
| `--max-steps` | int | preset | Can override preset |
| `--eval-every` | int | `200` | Training steps |
| `--save-every` | int | `500` | Training steps |
| `--keep-last-k` | int | `3` | Full checkpoint retention |
| `--learning-rate` | float | `5e-5` | AdamW |
| `--weight-decay` | float | `0.01` | AdamW |
| `--warmup-fraction` | float | `0.1` | Cosine scheduler |
| `--max-grad-norm` | float | `1.0` | Clip norm |
| `--lb-coef` | float | preset | `0.01` or `0.0` |
| `--z-coef` | float | `0.001` | Router z-loss |
| `--wandb-project` | str | `moe-emergence` | Tracking project |
| `--wandb-entity` | str | None | Optional |
| `--wandb-offline` | flag | False | Force offline mode |
| `--log-router-every` | int | `100` | Router metric cadence |
| `--num-experts` | int | `8` | MoE config |
| `--topk` | int | preset | `1` except `top2` |
| `--moe-layers` | list[int] | `8 9 10 11` | Target blocks |

## Preset Definitions
1. `shakedown`: `max_steps=100`, `size_mb=1`, `eval_every=50`, `save_every=50`, `balance_tokens=false`.
2. `dense`: `max_steps=5000`, `balance_tokens=true`, no MoE layer install, aux losses disabled.
3. `moe-main`: `max_steps=10000`, `balance_tokens=true`, top1, `lb=0.01`, `z=0.001`, `noise_std=0.1`.
4. `no-lb`: `max_steps=2000`, `balance_tokens=true`, top1, `lb=0.0`, `z=0.001`, **`noise_std=0.0`**, no annealing, collapse early-stop enabled.
5. `top2`: `max_steps=3000`, `balance_tokens=true`, top2, `lb=0.01`, `z=0.001`, `noise_std=0.1`.

**Note on `no-lb` noise:** Router noise is disabled (`noise_std=0.0`) in the `no-lb` preset to avoid confounding collapse analysis. With noise active, exploration delays collapse and makes the ablation result ambiguous. See debate 009, Issue 3.

## Data and Model Flow
1. Build tokenizer. Split texts via `split_texts_for_eval()` before packing. Build packed train/eval datasets via `moe_emergence/data.py` (eval dataset uses `balance_tokens=False`).
2. Build DataLoaders with `collate_packed`.
3. Load base model `GPT2LMHeadModel.from_pretrained("gpt2")`.
4. For MoE presets, call `install_moe_layers` from `moe_emergence/gpt2_moe.py`.
5. For dense preset, skip MoE install.
6. Use step-based loop with iterator reset on exhaustion.
7. Run eval at fixed step cadence.

## Loss and Optimization Contract
1. LM loss from `outputs.loss` with labels identical to input.
2. For MoE presets, collect aux outputs each step.
3. Compute layerwise LB and Z losses and average across active MoE layers.
4. Total loss: `lm + lb_coef * lb + z_coef * z`.
5. Support gradient accumulation with normalized loss.
6. Clip gradients before optimizer step.
7. Step scheduler and router training-step counters only on optimizer updates.
8. Scheduler: cosine with warmup. `num_training_steps` is always set to `max_steps` (not derived from dataset size). On resume, restore scheduler state dict directly; do not recompute progress.
9. Optimizer: AdamW.

## Tracking Contract
1. Attempt W&B online init first.
2. If online init fails, automatically fall back to offline.
3. Always continue training even if W&B is unavailable.
4. Log train losses every step.
5. Log router metrics every `log_router_every`.
6. Log eval loss and eval perplexity at eval cadence.
7. Persist minimal local JSONL/CSV metric log in run dir for safety.

## Checkpoint Contract
Checkpoint directory: `checkpoints/<run-name>/`

### Format Policy
1. Full resume checkpoints use `.pt` because optimizer/scheduler/RNG states are Python objects.
2. Model-only artifacts use `.safetensors` to avoid pickle-based loading risks.
3. `.pt` loading is allowed only for trusted local training artifacts created by this project.

### Full Resume Checkpoint
Filename: `ckpt-step-<n>.pt`
Required keys:
`format_version` (set to `1`), `step`, `preset`, `mode`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`, `python_random_state`, `numpy_random_state`, `torch_rng_state`, `cuda_rng_state_if_available`.
Resume must validate `format_version == 1` before loading.

### Model-Only Snapshot
Filenames:
1. `model-step-<n>.safetensors` (weights only)
2. `model-step-<n>.json` (metadata sidecar)

Model metadata JSON required keys:
`format_version`, `step`, `preset`, `mode`, `config`, `metrics_summary`.

### Best Model Selection
`best-model.safetensors` (+ `best-model.json`) is the model-only snapshot with the lowest aggregate eval loss (`eval/loss`). Updated at each eval checkpoint if current eval loss < best seen so far.

### Retention
1. Keep last `K` full checkpoints (`--keep-last-k`).
2. Always retain `best-model.safetensors`/`best-model.json` and `final-model.safetensors`/`final-model.json`.
3. Resume is permitted only from full checkpoints.
4. Resume must validate architecture/mode compatibility and `format_version` before loading.

## Collapse Detection Contract (`no-lb`)
1. Every 100 steps, compute expert fractions from aux outputs.
2. Collapse criterion: any expert fraction > `0.60` in any MoE layer.
3. Require criterion for 3 consecutive checks to reduce false triggers.
4. Stop run early and record reason in run summary.

## Mandatory Shakedown Gate
Before budgeted runs, both must pass:
1. Dense shakedown run.
2. MoE shakedown run.

Pass criteria:
1. No NaN/Inf in train or eval losses.
2. Checkpoints are written on schedule.
3. Resume from checkpoint succeeds and continues step count.
4. Tracking logs are produced.
5. Router metrics are produced in MoE mode.

## Run Order After Gate
1. Dense baseline.
2. MoE main run.
3. No-LB ablation.
4. Top2 directional run (optional budget dependent).

## Test Scenarios
1. Lint: `uv run ruff check moe_emergence`.
2. Dense shakedown command executes end-to-end.
3. MoE shakedown command executes end-to-end.
4. Resume test from step N to N+M on full checkpoint.
5. Device parity test on MPS and CUDA when both available.
6. No-LB collapse early-stop test.
7. Load model-only `.safetensors` snapshot in `moe_emergence/gpt2_inference.py`.

## Documentation Deliverables
1. Add Phase 4 decision log in `docs/decisions/`.
2. Add experiment logs for shakedown and each main run in `docs/experiments/`.
3. Include commit hash in every new doc entry.
4. Update `README.md` Phase 4 status after shakedown passes.

## Reproducibility Contract
1. At startup, call `seed_everything(seed)` which sets `random.seed`, `torch.manual_seed`, `numpy.random.seed`, and `torch.cuda.manual_seed_all` (when CUDA is available).
2. Full checkpoints include `numpy_random_state` alongside python and torch RNG states.
3. On resume, restore all RNG states before continuing training.
4. Scheduler and optimizer states are restored from checkpoint directly (no heuristic recomputation).

## OOM Fallback
Default micro-batch is `batch_size=2, grad_accum_steps=4` (effective batch = 8). If OOM occurs:
1. Reduce to `batch_size=1, grad_accum_steps=8` (same effective batch).
2. If still OOM, reduce `block_size` from 512 to 256 (halves sequence memory).

## Assumptions and Defaults
1. Dataset stack remains CodeParrot + MathQA + C4.
2. `--balance-tokens` is enabled for real training runs.
3. Shakedown uses reduced data/steps and can skip token balancing.
4. Import cleanup and eval formula cleanup are separate next-batch tasks.
5. Budget prioritization remains dense -> moe-main -> no-lb -> top2.
