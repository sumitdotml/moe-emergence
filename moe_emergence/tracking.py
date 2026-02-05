"""
W&B Experiment Tracking for MoE Training

Provides utilities for logging training metrics, router statistics,
and evaluation results. See docs/decisions/009-experiment-tracking.md.

Usage:
    from moe_emergence.tracking import init_run, log_step, log_eval, finish_run

    run = init_run(config)
    for step, batch in enumerate(loader):
        # ... training ...
        log_step(step, loss, domain_losses, aux_outputs)
    log_eval(step, eval_metrics)
    finish_run()
"""

import math
from typing import Optional

import torch

try:
    import wandb
except ImportError:
    wandb = None


def init_run(
    config: dict,
    project: str = "moe-emergence",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    offline: bool = False,
) -> Optional[object]:
    """
    Initialize a W&B run.

    Args:
        config: Hyperparameters dict (logged automatically)
        project: W&B project name
        entity: W&B username or team name (uses default if None)
        name: Run name (auto-generated if None)
        tags: List of tags for filtering runs
        offline: If True, log locally without syncing

    Returns:
        W&B run object, or None if wandb unavailable
    """
    if wandb is None:
        print("Warning: wandb not installed, skipping tracking")
        return None

    mode = "offline" if offline else "online"
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        config=config,
        mode=mode,
    )
    return run


def log_step(
    step: int,
    total_loss: float,
    lm_loss: float,
    lb_loss: float,
    z_loss: float,
    aux_outputs: Optional[list[dict]] = None,
    domain_losses: Optional[dict[str, float]] = None,
    learning_rate: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    log_router_every: int = 100,
) -> None:
    """
    Log training step metrics.

    Args:
        step: Current training step
        total_loss: Combined loss (LM + LB + Z)
        lm_loss: Language modeling loss
        lb_loss: Load balancing loss
        z_loss: Router z-loss
        aux_outputs: List of aux dicts from MoE layers (for router metrics)
        domain_losses: Per-domain LM losses {"code": x, "math": y, "prose": z}
        learning_rate: Current learning rate from scheduler
        tokens_per_sec: Training throughput
        log_router_every: Log router metrics every N steps (expensive)
    """
    if wandb is None or wandb.run is None:
        return

    metrics = {
        "train/loss": total_loss,
        "train/lm_loss": lm_loss,
        "train/lb_loss": lb_loss,
        "train/z_loss": z_loss,
        "step": step,
    }

    if learning_rate is not None:
        metrics["train/lr"] = learning_rate

    if tokens_per_sec is not None:
        metrics["perf/tokens_per_sec"] = tokens_per_sec

    if domain_losses:
        for domain, loss in domain_losses.items():
            metrics[f"train/loss_{domain}"] = loss

    if aux_outputs and step % log_router_every == 0:
        router_metrics = compute_router_metrics(aux_outputs)
        metrics.update(router_metrics)

    wandb.log(metrics, step=step)


def log_eval(
    step: int,
    eval_loss: float,
    domain_losses: Optional[dict[str, float]] = None,
    domain_perplexities: Optional[dict[str, float]] = None,
) -> None:
    """
    Log evaluation metrics.

    Args:
        step: Current training step
        eval_loss: Overall eval loss
        domain_losses: Per-domain eval losses
        domain_perplexities: Per-domain perplexities
    """
    if wandb is None or wandb.run is None:
        return

    metrics = {
        "eval/loss": eval_loss,
        "eval/perplexity": math.exp(eval_loss),
        "step": step,
    }

    if domain_losses:
        for domain, loss in domain_losses.items():
            metrics[f"eval/loss_{domain}"] = loss

    if domain_perplexities:
        for domain, ppl in domain_perplexities.items():
            metrics[f"eval/ppl_{domain}"] = ppl

    wandb.log(metrics, step=step)


def compute_router_metrics(aux_outputs: list[dict]) -> dict:
    """
    Compute router statistics from MoE layer aux outputs.

    Provides both per-layer and aggregated metrics as required by V3 spec.
    Logs two entropy measures:
    - entropy_per_token: average router confidence (from pre-computed aux["entropy"])
    - entropy_distribution: load balance across experts (computed from avg probs)

    Args:
        aux_outputs: List of dicts from collect_aux_outputs(), each containing:
            - layer_idx: int, the transformer layer index
            - router_probs_clean: [batch*seq, n_experts] clean routing probabilities
            - topk_indices: [batch*seq, k] selected expert indices
            - entropy: [batch*seq] per-token entropy (pre-computed on clean probs)

    Returns:
        Dict of router metrics for logging
    """
    metrics = {}

    if not aux_outputs:
        return metrics

    required_keys = ["layer_idx", "topk_indices", "entropy", "router_probs_clean"]
    n_experts = None
    all_utilizations = []
    all_per_token_entropies = []
    all_distribution_entropies = []

    for aux in aux_outputs:
        if not all(k in aux for k in required_keys):
            continue

        layer_idx = aux["layer_idx"]
        probs_clean = aux["router_probs_clean"]
        indices = aux["topk_indices"]
        per_token_entropy = aux["entropy"]

        n_experts = probs_clean.shape[1]

        # per-layer entropy: router confidence (pre-computed on clean probs)
        mean_per_token = per_token_entropy.mean().item()
        metrics[f"router/layer_{layer_idx}_entropy_per_token"] = mean_per_token
        all_per_token_entropies.append(mean_per_token)

        # per-layer entropy: distribution balance (computed from avg probs)
        avg_probs = probs_clean.mean(dim=0)
        dist_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum().item()
        metrics[f"router/layer_{layer_idx}_entropy_distribution"] = dist_entropy
        all_distribution_entropies.append(dist_entropy)

        # per-layer expert utilization
        layer_utils = []
        for i in range(n_experts):
            frac = (indices == i).float().mean().item()
            metrics[f"router/layer_{layer_idx}_expert_{i}_frac"] = frac
            layer_utils.append(frac)
        all_utilizations.append(layer_utils)

        # per-layer utilization std dev
        util_std = torch.tensor(layer_utils).std().item()
        metrics[f"router/layer_{layer_idx}_util_std"] = util_std

    # aggregated metrics across all layers
    if n_experts and all_utilizations:
        all_utils_tensor = torch.tensor(all_utilizations)
        avg_utils = all_utils_tensor.mean(dim=0)

        for i in range(n_experts):
            metrics[f"router/expert_{i}_frac"] = avg_utils[i].item()

        metrics["router/util_std"] = avg_utils.std().item()

        # aggregated entropy metrics
        metrics["router/entropy_per_token"] = sum(all_per_token_entropies) / len(
            all_per_token_entropies
        )
        metrics["router/entropy_distribution"] = sum(all_distribution_entropies) / len(
            all_distribution_entropies
        )

        max_entropy = math.log(n_experts)
        metrics["router/entropy_distribution_normalized"] = (
            metrics["router/entropy_distribution"] / max_entropy
        )

    return metrics


def log_expert_domain_affinity(
    step: int,
    affinity_matrix: torch.Tensor,
    domain_names: list[str] = ["code", "math", "prose"],
) -> None:
    """
    Log expert-domain affinity heatmap.

    Args:
        step: Current training step
        affinity_matrix: [n_experts, n_domains] routing fractions
        domain_names: Names for columns
    """
    if wandb is None or wandb.run is None:
        return

    n_experts = affinity_matrix.shape[0]
    expert_names = [f"expert_{i}" for i in range(n_experts)]

    table = wandb.Table(
        columns=["expert"] + domain_names,
        data=[
            [expert_names[i]] + affinity_matrix[i].tolist() for i in range(n_experts)
        ],
    )
    wandb.log({f"affinity/table_step_{step}": table}, step=step)


def log_gpu_memory(step: int) -> None:
    """Log current GPU memory usage."""
    if wandb is None or wandb.run is None:
        return

    if not torch.cuda.is_available():
        return

    metrics = {
        "perf/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "perf/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "perf/gpu_max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    wandb.log(metrics, step=step)


def finish_run() -> None:
    """Finish the W&B run."""
    if wandb is not None and wandb.run is not None:
        wandb.finish()
