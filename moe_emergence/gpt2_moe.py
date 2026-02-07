"""
MoE Wrapper for GPT-2 Integration

This module provides a drop-in replacement for GPT2MLP that routes tokens
through multiple expert copies of the original MLP.

Usage:
    from transformers import GPT2LMHeadModel
    from gpt2_moe import install_moe_layers

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model, moe_modules = install_moe_layers(model, moe_layers=[8, 9, 10, 11])
"""

import copy
from typing import NamedTuple, Optional

import torch
from torch import Tensor
import torch.nn as nn

from moe_emergence.moe import Router, compute_load_balance_loss, compute_z_loss


class MoEWrapperOutput(NamedTuple):
    """Auxiliary outputs stored after each forward pass for loss computation."""

    router_probs: Tensor  # [n_tokens, n_experts] - for load balancing loss (post-noise)
    router_probs_clean: (
        Tensor  # [n_tokens, n_experts] - for entropy logging (pre-noise)
    )
    router_logits: Tensor  # [n_tokens, n_experts] - for z-loss
    topk_indices: Tensor  # [n_tokens, topk] - which experts were selected
    topk_weights: Tensor  # [n_tokens, topk] - routing weights
    entropy: Tensor  # [n_tokens] - router entropy from clean probs


class MoEWrapper(nn.Module):
    """
    Drop-in replacement for GPT2MLP that routes to multiple expert copies.

    This wrapper:
    1. has the same interface as GPT2MLP (takes hidden_states, returns hidden_states)
    2. creates experts as deepcopies of the original MLP (warm-start)
    3. adds small noise for symmetry breaking
    4. stores auxiliary outputs for loss computation

    By using deepcopy, each expert starts as an exact copy of the pretrained MLP.
    At step 0, the model behaves identically to the pretrained GPT-2.
    As training proceeds, experts gradually specialize.

    Args:
        original_mlp: The GPT2MLP module to replace (will be deepcopied)
        hidden_dim: Hidden dimension (get from model.config.n_embd, e.g. 768 for GPT-2 small)
        n_experts: Number of expert copies to create (e.g. 8 in Mixtral 8x7B)
        topk: Number of experts to route each token to (e.g. 2 in Mixtral 8x7B)
        noise_std: Router noise for exploration (annealed during training)
        perturbation_std: Noise added to expert weights for symmetry breaking

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> block = model.transformer.h[8]
        >>> moe = MoEWrapper(block.mlp, hidden_dim=768, n_experts=8, topk=1)
        >>> block.mlp = moe  # gotta do this to replace the MLP with MoE
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        n_experts: int = 8,
        topk: int = 1,
        noise_std: float = 0.1,
        perturbation_std: float = 1e-3,
    ) -> None:
        super().__init__()
        assert topk <= n_experts, f"topk ({topk}) cannot exceed n_experts ({n_experts})"
        self.n_experts = n_experts
        self.topk = topk

        self.router = Router(
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
        )

        # :::warm-start:::
        # creating experts as EXACT COPIES of original MLP. at step 0, any
        # expert produces the same output as the original MLP would have
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = copy.deepcopy(original_mlp)
            # :::symmetry breaking:::
            # adding small perturbation for symmetry breaking. without this, all experts would
            # receive identical gradients
            # scale: 1e-3 * parameter std (NOT norm, see project design doc: MOE-PROJECT-DESIGN-V3.md)
            with torch.no_grad():
                for param in expert.parameters():
                    noise = torch.randn_like(param) * param.std() * perturbation_std
                    param.add_(noise)

            self.experts.append(expert)

        # :::storage for auxiliary outputs:::
        # retrieved after forward pass for loss computation
        self.last_aux: Optional[MoEWrapperOutput] = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass matching GPT2MLP interface.

        GPT2MLP.forward signature: (hidden_states) -> hidden_states
        We must match this exactly for drop-in replacement.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            output: [batch, seq_len, hidden_dim] (same shape as input)
        """
        batch, seq_len, hidden_dim = hidden_states.shape

        hidden_flat = hidden_states.reshape(
            -1, hidden_dim
        )  # [batch * seq_len, hidden_dim]

        (
            topk_weights,
            topk_indices,
            router_probs,
            router_probs_clean,
            router_logits,
            entropy,
        ) = self.router(hidden_states)

        # dispatching tokens to experts and combine outputs
        output_flat = self._dispatch_experts(hidden_flat, topk_weights, topk_indices)

        # storing auxiliary outputs for loss computation
        # these are retrieved by the training loop after forward pass
        self.last_aux = MoEWrapperOutput(
            router_probs=router_probs,
            router_probs_clean=router_probs_clean,
            router_logits=router_logits,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            entropy=entropy,
        )

        # reshaping back to original shape
        return output_flat.reshape(batch, seq_len, hidden_dim)

    def _dispatch_experts(
        self, hidden_flat: Tensor, topk_weights: Tensor, topk_indices: Tensor
    ) -> Tensor:
        """
        Dispatch tokens to their selected experts and combine outputs.

        Args:
            hidden_flat: [n_tokens, hidden_dim]
            topk_weights: [n_tokens, topk]
            topk_indices: [n_tokens, topk]

        Returns:
            output: [n_tokens, hidden_dim]
        """
        results = torch.zeros_like(hidden_flat)

        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(topk_indices == i)

            if len(token_idx) == 0:
                continue

            expert_input = hidden_flat[token_idx]  # [num_selected, hidden_dim]
            weights = topk_weights[token_idx, topk_idx].unsqueeze(
                -1
            )  # [num_selected, 1]

            expert_output = expert(expert_input)  # [num_selected, hidden_dim]
            results[token_idx] += weights * expert_output

        return results


# :::GPT-2 Surgery: Installing MoE Layers:::


def install_moe_layers(
    model,  # GPT2LMHeadModel
    moe_layers: list[int] = [8, 9, 10, 11],
    n_experts: int = 8,
    topk: int = 1,
    noise_std: float = 0.1,
) -> tuple:
    """
    Replaces specified GPT-2 MLP layers with MoE wrappers.

    This is "model surgery": we swap out the MLP modules but leave
    everything else (attention, layer norms, residual connections) untouched.
    HuggingFace's forward pass works unchanged.

    Args:
        model: GPT2LMHeadModel from HuggingFace
        moe_layers: Which layer indices to convert to MoE (default: last 4)
        n_experts: Number of experts per MoE layer
        topk: Number of experts to route each token to
        noise_std: Router noise for exploration

    Returns:
        model: Modified model (in-place)
        moe_modules: Dict mapping layer_idx -> MoEWrapper (for aux retrieval)

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> model, moe_modules = install_moe_layers(model, moe_layers=[8, 9, 10, 11])
        >>> # Forward pass works as normal
        >>> outputs = model(input_ids)
        >>> # Retrieve aux outputs for loss computation
        >>> for layer_idx, moe in moe_modules.items():
        >>>     aux = moe.last_aux
    """
    moe_modules = {}
    hidden_dim = model.config.n_embd  # 768 for GPT-2 small

    for layer_idx in moe_layers:
        block = model.transformer.h[layer_idx]
        original_mlp = block.mlp

        # MoE wrapper where experts are initialized as copies of original MLP
        moe = MoEWrapper(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
        )

        # replacing the original MLP with the MoE wrapper
        block.mlp = moe

        # storing reference for auxiliary outputs retrieval
        moe_modules[layer_idx] = moe

        print(
            f"Installed MoE at layer {layer_idx}: "
            f"{n_experts} experts, top-{topk} routing"
        )

    return model, moe_modules


def collect_aux_outputs(moe_modules: dict) -> list[dict]:
    """
    Collects auxiliary outputs from all MoE layers after a forward pass.

    This is called after `model(input_ids)` to get routing stats for loss computation.

    Args:
        moe_modules: Dict from `install_moe_layers()`

    Returns:
        List of dicts with routing stats per layer (probs, probs_clean, logits, indices, weights, entropy)
    """
    aux_outputs = []

    for layer_idx, moe in moe_modules.items():
        if moe.last_aux is not None:
            aux_outputs.append(
                {
                    "layer_idx": layer_idx,
                    "router_probs": moe.last_aux.router_probs,
                    "router_probs_clean": moe.last_aux.router_probs_clean,
                    "router_logits": moe.last_aux.router_logits,
                    "topk_indices": moe.last_aux.topk_indices,
                    "topk_weights": moe.last_aux.topk_weights,
                    "entropy": moe.last_aux.entropy,
                }
            )

    return aux_outputs


# :::Quick Verification:::

if __name__ == "__main__":
    # quick test without actually loading GPT-2 to avoid large download
    print("Testing MoEWrapper with a dummy MLP...")

    class DummyMLP(nn.Module):
        def __init__(self, hidden_dim=768, ffn_dim=3072):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, ffn_dim)
            self.fc2 = nn.Linear(ffn_dim, hidden_dim)
            self.act = nn.GELU()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    dummy_mlp = DummyMLP()
    moe = MoEWrapper(dummy_mlp, hidden_dim=768, n_experts=8, topk=1)

    x = torch.randn(2, 10, 768)  # [batch=2, seq=10, hidden=768]
    output = moe(x)

    assert output.shape == x.shape, (
        f"Output shape {output.shape} does not match input shape {x.shape}"
    )
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    aux = moe.last_aux
    assert aux is not None, "Aux outputs are not populated"

    n_tokens = x.shape[0] * x.shape[1]
    n_experts = moe.n_experts
    topk = moe.topk

    assert aux.router_probs.shape == (n_tokens, n_experts)
    assert aux.router_probs_clean.shape == (n_tokens, n_experts)
    assert aux.router_logits.shape == (n_tokens, n_experts)
    assert aux.topk_indices.shape == (n_tokens, topk)
    assert aux.topk_weights.shape == (n_tokens, topk)
    assert aux.entropy.shape == (n_tokens,)
    assert aux.topk_indices.min() >= 0
    assert aux.topk_indices.max() < n_experts
    assert torch.allclose(aux.topk_weights.sum(dim=-1), torch.ones(n_tokens), atol=1e-6)

    # loss computation
    lb_loss = compute_load_balance_loss(aux.router_probs, aux.topk_indices, n_experts)
    z_loss = compute_z_loss(aux.router_logits)
    assert lb_loss.ndim == 0, "Load balance loss is not scalar"
    assert z_loss.ndim == 0, "Z-loss is not scalar"
    assert torch.isfinite(lb_loss), "Load balance loss is not finite"
    assert torch.isfinite(z_loss), "Z-loss is not finite"
    print(f"Load balance loss: {lb_loss.item():.4f} (target ~1.0)")
    print(f"Z-loss: {z_loss.item():.4f}")

    loss = output.mean()
    loss.backward()
    router_grad = moe.router.gate.weight.grad
    assert router_grad is not None, "Router has no gradient"
    assert router_grad.abs().sum() > 0, "Router gradient is zero"
