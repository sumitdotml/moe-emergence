"""
Training script for MoE-GPT-2 experiments.

Supports dense baseline and MoE training paths in one script.
See docs/project-design/PHASE-4-TRAINING-PLAN.md for the full spec.

Usage:
    # Shakedown (quick sanity check)
    uv run python -m moe_emergence.train --preset shakedown --run-name shake-dense
    uv run python -m moe_emergence.train --preset shakedown --run-name shake-moe --moe-layers 8 9 10 11

    # Dense baseline
    uv run python -m moe_emergence.train --preset dense --run-name dense-baseline

    # MoE main run
    uv run python -m moe_emergence.train --preset moe-main --run-name moe-main

    # No load-balancing ablation (collapse detection enabled)
    uv run python -m moe_emergence.train --preset no-lb --run-name no-lb-ablation

    # Resume from checkpoint
    uv run python -m moe_emergence.train --preset moe-main --run-name moe-main --resume checkpoints/moe-main/ckpt-step-500.pt
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import save_file as save_safetensors
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from moe_emergence import tracking
from moe_emergence.data import (
    PackedMixedDomainDataset,
    collate_packed,
    load_code_data,
    load_math_data,
    load_prose_data,
    split_texts_for_eval,
)
from moe_emergence.gpt2_moe import (
    collect_aux_outputs,
    install_moe_layers,
)
from moe_emergence.moe import compute_load_balance_loss, compute_z_loss

REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / ".cache"


# ::: Preset definitions :::

PRESETS = {
    "shakedown": dict(
        max_steps=100,
        size_mb=1.0,
        eval_every=50,
        save_every=50,
        balance_tokens=False,
        moe_layers=[],
        lb_coef=0.01,
        z_coef=0.001,
        topk=1,
        noise_std=0.1,
    ),
    "dense": dict(
        max_steps=5000,
        size_mb=10.0,
        eval_every=200,
        save_every=500,
        balance_tokens=True,
        moe_layers=[],
        lb_coef=0.0,
        z_coef=0.0,
        topk=1,
        noise_std=0.0,
    ),
    "moe-main": dict(
        max_steps=10000,
        size_mb=10.0,
        eval_every=200,
        save_every=500,
        balance_tokens=True,
        moe_layers=[8, 9, 10, 11],
        lb_coef=0.01,
        z_coef=0.001,
        topk=1,
        noise_std=0.1,
    ),
    "no-lb": dict(
        max_steps=2000,
        size_mb=10.0,
        eval_every=200,
        save_every=500,
        balance_tokens=True,
        moe_layers=[8, 9, 10, 11],
        lb_coef=0.0,
        z_coef=0.001,
        topk=1,
        noise_std=0.0,
    ),
    "top2": dict(
        max_steps=3000,
        size_mb=10.0,
        eval_every=200,
        save_every=500,
        balance_tokens=True,
        moe_layers=[8, 9, 10, 11],
        lb_coef=0.01,
        z_coef=0.001,
        topk=2,
        noise_std=0.1,
    ),
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

FORMAT_VERSION = 1


def save_full_checkpoint(
    path: Path,
    step: int,
    preset: str,
    mode: str,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler,
    config: dict,
) -> None:
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "preset": preset,
        "mode": mode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_if_available": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(checkpoint, path)


def save_model_snapshot(
    path: Path,
    step: int,
    preset: str,
    mode: str,
    model: torch.nn.Module,
    config: dict,
    metrics_summary: dict,
) -> None:
    """Save model weights as .safetensors with a .json metadata sidecar."""
    # cloning to break shared storage from GPT-2 tied weights
    st_path = path.with_suffix(".safetensors")
    save_safetensors(
        {k: v.clone() for k, v in model.state_dict().items()}, str(st_path)
    )

    meta_path = path.with_suffix(".json")
    meta = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "preset": preset,
        "mode": mode,
        "config": config,
        "metrics_summary": metrics_summary,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    device,
    *,
    expected_mode: str,
    expected_config: dict,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version={ckpt.get('format_version')}, expected {FORMAT_VERSION}"
        )
    ckpt_mode = ckpt.get("mode", "unknown")
    if ckpt_mode != expected_mode:
        raise ValueError(
            f"Checkpoint mode='{ckpt_mode}' does not match current mode='{expected_mode}'"
        )
    ckpt_config = ckpt.get("config", {})
    for key in ("topk", "n_experts", "moe_layers"):
        ckpt_val = ckpt_config.get(key)
        cur_val = expected_config.get(key)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            raise ValueError(
                f"Checkpoint {key}={ckpt_val} does not match current {key}={cur_val}"
            )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    random.setstate(ckpt["python_random_state"])
    np.random.set_state(ckpt["numpy_random_state"])
    torch.random.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt["cuda_rng_state_if_available"] is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_if_available"])

    return ckpt["step"] + 1, ckpt.get("config", {}), ckpt.get("preset", "unknown")


def enforce_retention(run_dir: Path, keep_last_k: int) -> None:
    """Delete old full checkpoints, keeping only the last K."""
    ckpts = sorted(run_dir.glob("ckpt-step-*.pt"), key=os.path.getmtime)
    while len(ckpts) > keep_last_k:
        ckpts.pop(0).unlink()


# ---------------------------------------------------------------------------
# Collapse detection (no-lb preset)
# ---------------------------------------------------------------------------


def check_collapse(
    aux_outputs: list[dict],
    n_experts: int,
    threshold: float = 0.60,
) -> tuple[bool, str]:
    """Returns (collapsed, reason)."""
    for aux in aux_outputs:
        indices = aux["topk_indices"]
        counts = torch.bincount(indices.view(-1), minlength=n_experts).float()
        fracs = counts / counts.sum()
        max_frac = fracs.max().item()
        if max_frac > threshold:
            dominant = fracs.argmax().item()
            reason = (
                f"layer {aux['layer_idx']}: expert {dominant} "
                f"handles {max_frac:.1%} of tokens"
            )
            return True, reason
    return False, ""


# ---------------------------------------------------------------------------
# Local metric logger (JSONL safety net)
# ---------------------------------------------------------------------------


class LocalMetricLogger:
    def __init__(self, run_dir: Path):
        self.jsonl_path = run_dir / "metrics.jsonl"

    def log(self, metrics: dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


# ---------------------------------------------------------------------------
# Infinite data iterator
# ---------------------------------------------------------------------------


def infinite_loader(loader: DataLoader):
    """Yields batches forever, resetting when exhausted."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    moe_modules: dict,
    device: torch.device,
    lb_coef: float,
    z_coef: float,
    n_experts: int,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_lm_loss = 0.0
    total_lb_loss = 0.0
    total_z_loss = 0.0
    n_batches = 0
    domain_losses: dict[str, list[float]] = {}

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        domains = batch["domains"]

        outputs = model(input_ids=input_ids, labels=input_ids)
        lm_loss = outputs.loss.item()

        lb_loss_val = 0.0
        z_loss_val = 0.0
        if moe_modules:
            aux_outputs = collect_aux_outputs(moe_modules)
            for aux in aux_outputs:
                lb_loss_val += compute_load_balance_loss(
                    aux["router_probs"], aux["topk_indices"], n_experts
                ).item()
                z_loss_val += compute_z_loss(aux["router_logits"]).item()
            n_layers = len(aux_outputs) or 1
            lb_loss_val /= n_layers
            z_loss_val /= n_layers

        step_loss = lm_loss + lb_coef * lb_loss_val + z_coef * z_loss_val
        total_loss += step_loss
        total_lm_loss += lm_loss
        total_lb_loss += lb_loss_val
        total_z_loss += z_loss_val
        n_batches += 1

        # per-domain tracking (batch-level: each batch has mixed domains)
        for d in domains:
            domain_losses.setdefault(d, []).append(lm_loss)

    model.train()

    n = max(n_batches, 1)
    avg_lm = total_lm_loss / n
    result = {
        "eval/loss": total_loss / n,
        "eval/lm_loss": avg_lm,
        "eval/lb_loss": total_lb_loss / n,
        "eval/z_loss": total_z_loss / n,
        "eval/perplexity": math.exp(min(avg_lm, 20)),  # clamp for safety
    }

    domain_ppl = {}
    for d, losses in domain_losses.items():
        avg = sum(losses) / len(losses)
        result[f"eval/loss_{d}"] = avg
        ppl = math.exp(min(avg, 20))
        result[f"eval/ppl_{d}"] = ppl
        domain_ppl[d] = ppl

    return result


# ::: Main training function :::


def train(args: argparse.Namespace) -> None:
    # ::: resolve preset :::
    preset_cfg = PRESETS[args.preset].copy()

    # CLI overrides
    if args.max_steps is not None:
        preset_cfg["max_steps"] = args.max_steps
    if args.moe_layers is not None:
        preset_cfg["moe_layers"] = args.moe_layers
    if args.balance_tokens:
        preset_cfg["balance_tokens"] = True
    if args.lb_coef is not None:
        preset_cfg["lb_coef"] = args.lb_coef
    if args.z_coef is not None:
        preset_cfg["z_coef"] = args.z_coef
    if args.topk is not None:
        preset_cfg["topk"] = args.topk
    if args.size_mb is not None:
        preset_cfg["size_mb"] = args.size_mb
    if args.eval_every is not None:
        preset_cfg["eval_every"] = args.eval_every
    if args.save_every is not None:
        preset_cfg["save_every"] = args.save_every

    moe_layers = preset_cfg["moe_layers"]
    is_moe = len(moe_layers) > 0
    mode = "moe" if is_moe else "dense"
    max_steps = preset_cfg["max_steps"]
    lb_coef = preset_cfg["lb_coef"]
    z_coef = preset_cfg["z_coef"]
    topk = preset_cfg["topk"]
    noise_std = preset_cfg["noise_std"]
    n_experts = args.num_experts
    collapse_early_stop = args.preset == "no-lb"

    # ::: config dict for logging/checkpointing :::
    config = {
        "preset": args.preset,
        "mode": mode,
        "run_name": args.run_name,
        "seed": args.seed,
        "max_steps": max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "block_size": args.block_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "max_grad_norm": args.max_grad_norm,
        "lb_coef": lb_coef,
        "z_coef": z_coef,
        "n_experts": n_experts,
        "topk": topk,
        "noise_std": noise_std,
        "moe_layers": moe_layers,
        "size_mb": preset_cfg["size_mb"],
        "balance_tokens": preset_cfg["balance_tokens"],
        "eval_every": preset_cfg["eval_every"],
        "save_every": preset_cfg["save_every"],
        "collapse_early_stop": collapse_early_stop,
    }

    # ::: setup :::
    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")
    print(f"Preset: {args.preset} | Mode: {mode} | Max steps: {max_steps}")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    local_logger = LocalMetricLogger(run_dir)

    # ::: W&B init :::
    tags = [args.preset, mode]
    try:
        tracking.init_run(
            config=config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            tags=tags,
            offline=args.wandb_offline,
        )
    except Exception as e:
        print(f"W&B online init failed ({e}), falling back to offline")
        try:
            tracking.init_run(
                config=config,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                tags=tags,
                offline=True,
            )
        except Exception:
            print("W&B unavailable, continuing without tracking")

    # ::: tokenizer :::
    hf_cache = str(CACHE_DIR / "huggingface")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=hf_cache)

    # ::: data :::
    print(f"\nLoading datasets ({preset_cfg['size_mb']}MB per domain)...")
    code_texts, _ = load_code_data(max_size_mb=preset_cfg["size_mb"])
    math_texts, _ = load_math_data(max_size_mb=preset_cfg["size_mb"])
    prose_texts, _ = load_prose_data(max_size_mb=preset_cfg["size_mb"])

    code_train, code_eval = split_texts_for_eval(code_texts, args.seed)
    math_train, math_eval = split_texts_for_eval(math_texts, args.seed)
    prose_train, prose_eval = split_texts_for_eval(prose_texts, args.seed)

    print("Building train dataset...")
    train_dataset = PackedMixedDomainDataset(
        code_train,
        math_train,
        prose_train,
        tokenizer,
        block_size=args.block_size,
        balance_tokens=preset_cfg["balance_tokens"],
        seed=args.seed,
    )
    print("Building eval dataset...")
    eval_dataset = PackedMixedDomainDataset(
        code_eval,
        math_eval,
        prose_eval,
        tokenizer,
        block_size=args.block_size,
        balance_tokens=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_packed,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_packed,
    )

    print(f"Train: {len(train_dataset)} blocks | Eval: {len(eval_dataset)} blocks")

    # ::: model :::
    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=hf_cache)

    moe_modules = {}
    if is_moe:
        model, moe_modules = install_moe_layers(
            model,
            moe_layers=moe_layers,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
        )

    model = model.to(device)
    model.train()

    # ::: optimizer + scheduler :::
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * args.warmup_fraction),
        num_training_steps=max_steps,
    )

    # ::: noise annealing :::
    if is_moe:
        for moe in moe_modules.values():
            moe.router.set_noise_annealing(max_steps, anneal_fraction=0.25)

    # ::: resume :::
    start_step = 0
    best_eval_loss = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        start_step, _, resumed_preset = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
            expected_mode=mode,
            expected_config=config,
        )
        if is_moe:
            for moe in moe_modules.values():
                moe.router.training_step.fill_(start_step)
        print(f"Resumed at step {start_step} (preset: {resumed_preset})")

    # checking for existing best model (metadata is in .json sidecar)
    best_model_stem = run_dir / "best-model"
    best_model_meta = best_model_stem.with_suffix(".json")
    if best_model_meta.exists():
        with open(best_model_meta) as f:
            best_meta = json.load(f)
        best_eval_loss = best_meta.get("metrics_summary", {}).get(
            "eval_loss", float("inf")
        )
        print(f"Existing best model at eval_loss={best_eval_loss:.4f}")

    # ::: training loop :::
    print(f"\n{'=' * 60}")
    print(f"Starting training: steps {start_step} → {max_steps}")
    print(
        f"Effective batch: {args.batch_size} × {args.grad_accum_steps} = "
        f"{args.batch_size * args.grad_accum_steps}"
    )
    print(f"{'=' * 60}\n")

    data_iter = infinite_loader(train_loader)
    optimizer.zero_grad()
    collapse_counter = 0
    tokens_processed = 0
    step_start_time = time.time()

    for step in range(start_step, max_steps):
        # ::: accumulation loop :::
        accum_lm = 0.0
        accum_lb = 0.0
        accum_z = 0.0
        last_aux = None

        for _micro in range(args.grad_accum_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)

            outputs = model(input_ids=input_ids, labels=input_ids)
            lm_loss = outputs.loss / args.grad_accum_steps

            lb_loss_scaled = torch.tensor(0.0, device=device)
            z_loss_scaled = torch.tensor(0.0, device=device)

            if is_moe:
                aux_outputs = collect_aux_outputs(moe_modules)
                last_aux = aux_outputs

                total_lb = torch.tensor(0.0, device=device)
                total_z = torch.tensor(0.0, device=device)
                for aux in aux_outputs:
                    total_lb = total_lb + compute_load_balance_loss(
                        aux["router_probs"],
                        aux["topk_indices"],
                        n_experts,
                    )
                    total_z = total_z + compute_z_loss(aux["router_logits"])
                n_layers = len(aux_outputs) or 1
                total_lb = total_lb / n_layers
                total_z = total_z / n_layers

                lb_loss_scaled = lb_coef * total_lb / args.grad_accum_steps
                z_loss_scaled = z_coef * total_z / args.grad_accum_steps

            loss = lm_loss + lb_loss_scaled + z_loss_scaled
            loss.backward()

            accum_lm += lm_loss.item() * args.grad_accum_steps
            if is_moe:
                accum_lb += total_lb.item()
                accum_z += total_z.item()

            tokens_processed += input_ids.numel()

        # ::: optimizer step :::
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if is_moe:
            for moe in moe_modules.values():
                moe.router.step()

        # ::: averaged metrics for this step :::
        avg_lm = accum_lm / args.grad_accum_steps
        avg_lb = accum_lb / args.grad_accum_steps
        avg_z = accum_z / args.grad_accum_steps
        total_loss = avg_lm + lb_coef * avg_lb + z_coef * avg_z

        # ::: logging :::::
        elapsed = time.time() - step_start_time
        tps = tokens_processed / elapsed if elapsed > 0 else 0

        current_lr = scheduler.get_last_lr()[0]

        # logging every step to W&B
        tracking.log_step(
            step=step,
            total_loss=total_loss,
            lm_loss=avg_lm,
            lb_loss=avg_lb,
            z_loss=avg_z,
            aux_outputs=last_aux if is_moe else None,
            learning_rate=current_lr,
            tokens_per_sec=tps,
            log_router_every=args.log_router_every,
        )

        # local JSONL
        local_logger.log(
            {
                "step": step,
                "train/loss": total_loss,
                "train/lm_loss": avg_lm,
                "train/lb_loss": avg_lb,
                "train/z_loss": avg_z,
                "train/lr": current_lr,
                "perf/tokens_per_sec": tps,
            }
        )

        # console
        if step % 10 == 0 or step == start_step:
            print(
                f"step {step:>5d}/{max_steps} | "
                f"loss {total_loss:.4f} | lm {avg_lm:.4f} | "
                f"lb {avg_lb:.4f} | z {avg_z:.4f} | "
                f"lr {current_lr:.2e} | {tps:.0f} tok/s"
            )

        # ::: collapse detection (no-lb) :::
        if collapse_early_stop and is_moe and last_aux and step % 100 == 0:
            collapsed, reason = check_collapse(last_aux, n_experts)
            if collapsed:
                collapse_counter += 1
                print(f"Collapse warning ({collapse_counter}/3): {reason}")
                if collapse_counter >= 3:
                    print(f"\nCollapse confirmed at step {step}. Stopping early.")
                    summary = {
                        "stopped_reason": "collapse",
                        "collapse_step": step,
                        "collapse_detail": reason,
                    }
                    with open(run_dir / "run_summary.json", "w") as f:
                        json.dump(summary, f, indent=2)
                    break
            else:
                collapse_counter = 0

        # ::: eval :::
        if step > 0 and step % preset_cfg["eval_every"] == 0:
            print(f"\n--- Eval at step {step} ---")
            eval_results = run_eval(
                model,
                eval_loader,
                moe_modules,
                device,
                lb_coef,
                z_coef,
                n_experts,
            )
            for k, v in eval_results.items():
                print(f"  {k}: {v:.4f}")
            print()

            # W&B eval logging
            domain_losses = {
                k.replace("eval/loss_", ""): v
                for k, v in eval_results.items()
                if k.startswith("eval/loss_") and k != "eval/loss"
            }
            domain_ppls = {
                k.replace("eval/ppl_", ""): v
                for k, v in eval_results.items()
                if k.startswith("eval/ppl_")
            }
            tracking.log_eval(
                step=step,
                eval_loss=eval_results["eval/loss"],
                domain_losses=domain_losses or None,
                domain_perplexities=domain_ppls or None,
            )

            # local JSONL
            local_logger.log({"step": step, **eval_results})

            # best model tracking
            eval_loss = eval_results["eval/loss"]
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_model_snapshot(
                    best_model_stem,
                    step,
                    args.preset,
                    mode,
                    model,
                    config,
                    {
                        "eval_loss": eval_loss,
                        "eval_perplexity": eval_results["eval/perplexity"],
                    },
                )
                print(f"  New best model saved (eval_loss={eval_loss:.4f})")

        # ::: checkpointing :::
        if step > 0 and step % preset_cfg["save_every"] == 0:
            ckpt_path = run_dir / f"ckpt-step-{step}.pt"
            save_full_checkpoint(
                ckpt_path,
                step,
                args.preset,
                mode,
                model,
                optimizer,
                scheduler,
                config,
            )
            print(f"Saved checkpoint: {ckpt_path}")

            snap_path = run_dir / f"model-step-{step}"
            save_model_snapshot(
                snap_path,
                step,
                args.preset,
                mode,
                model,
                config,
                {"train_loss": total_loss, "lm_loss": avg_lm},
            )

            enforce_retention(run_dir, args.keep_last_k)

        # resetting timing
        tokens_processed = 0
        step_start_time = time.time()

    # ::: final save :::
    if start_step >= max_steps:
        print(
            f"\nRun already complete (resumed at step {start_step}, max_steps={max_steps})."
        )
    else:
        # step is the last completed step index (max_steps - 1)
        final_model_stem = run_dir / "final-model"
        save_model_snapshot(
            final_model_stem,
            step,
            args.preset,
            mode,
            model,
            config,
            {"train_loss": total_loss, "lm_loss": avg_lm},
        )
        print(f"\nTraining complete at step {step}.")
        print(f"Final model saved: {final_model_stem}.safetensors")

        final_ckpt_path = run_dir / f"ckpt-step-{step}.pt"
        if not final_ckpt_path.exists():
            save_full_checkpoint(
                final_ckpt_path,
                step,
                args.preset,
                mode,
                model,
                optimizer,
                scheduler,
                config,
            )
            enforce_retention(run_dir, args.keep_last_k)

    last_step = step if start_step < max_steps else start_step - 1
    summary = {
        "preset": args.preset,
        "mode": mode,
        "final_step": last_step,
        "max_steps": max_steps,
        "best_eval_loss": best_eval_loss,
        "final_train_loss": total_loss if start_step < max_steps else None,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    tracking.finish_run()
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE-GPT-2 Training")
    p.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to full checkpoint")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)

    # data
    p.add_argument("--size-mb", type=float, default=None)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--balance-tokens", action="store_true")

    # training
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--keep-last-k", type=int, default=3)

    # optimizer
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # MoE
    p.add_argument("--lb-coef", type=float, default=None)
    p.add_argument("--z-coef", type=float, default=None)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--moe-layers", type=int, nargs="+", default=None)

    # tracking
    p.add_argument("--wandb-project", default="moe-emergence")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--log-router-every", type=int, default=100)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
