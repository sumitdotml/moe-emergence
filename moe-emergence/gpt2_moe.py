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
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, NamedTuple

# Reuse the Router from our standalone MoE implementation
from moe import Router


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
    1. Has the SAME interface as GPT2MLP (takes hidden_states, returns hidden_states)
    2. Creates experts as deepcopies of the original MLP (warm-start)
    3. Adds small noise for symmetry breaking
    4. Stores auxiliary outputs for loss computation

    The key insight: by using deepcopy, each expert starts as an exact copy of
    the pretrained MLP. At step 0, the model behaves identically to the original
    GPT-2. As training proceeds, experts gradually specialize.

    Args:
        original_mlp: The GPT2MLP module to replace (will be deepcopied)
        hidden_dim: Hidden dimension (get from model.config.n_embd)
        n_experts: Number of expert copies to create
        topk: Number of experts to route each token to
        noise_std: Router noise for exploration (annealed during training)
        perturbation_std: Noise added to expert weights for symmetry breaking

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> block = model.transformer.h[8]
        >>> moe = MoEWrapper(block.mlp, hidden_dim=768, n_experts=8, topk=1)
        >>> block.mlp = moe  # Replace the MLP with MoE
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
        self.n_experts = n_experts
        self.topk = topk

        # Router decides which experts handle each token
        self.router = Router(
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
        )

        # Create experts as EXACT COPIES of original MLP
        # This is the warm-start: at step 0, any expert produces the same output
        # as the original MLP would have.
        self.experts = nn.ModuleList()
        for i in range(n_experts):
            expert = copy.deepcopy(original_mlp)

            # Add small perturbation for symmetry breaking
            # Without this, all experts would receive identical gradients
            # Scale: 1e-3 * parameter std (NOT norm — see project design doc)
            with torch.no_grad():
                for param in expert.parameters():
                    noise = torch.randn_like(param) * param.std() * perturbation_std
                    param.add_(noise)

            self.experts.append(expert)

        # Storage for auxiliary outputs (retrieved after forward for loss computation)
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

        # Flatten for routing: [batch * seq_len, hidden_dim]
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        # Get routing decisions
        (
            topk_weights,
            topk_indices,
            router_probs,
            router_probs_clean,
            router_logits,
            entropy,
        ) = self.router(hidden_states)

        # Dispatch tokens to experts and combine outputs
        output_flat = self._dispatch_experts(hidden_flat, topk_weights, topk_indices)

        # Store auxiliary outputs for loss computation
        # These are retrieved by the training loop after forward pass
        self.last_aux = MoEWrapperOutput(
            router_probs=router_probs,
            router_probs_clean=router_probs_clean,
            router_logits=router_logits,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            entropy=entropy,
        )

        # Reshape back to original shape
        return output_flat.view(batch, seq_len, hidden_dim)

    def _dispatch_experts(
        self, hidden_flat: Tensor, topk_weights: Tensor, topk_indices: Tensor
    ) -> Tensor:
        """
        Dispatch tokens to their selected experts and combine outputs.

        This uses the same loop-based approach as our standalone MoE.
        Not the fastest, but clear and correct.

        Args:
            hidden_flat: [n_tokens, hidden_dim]
            topk_weights: [n_tokens, topk]
            topk_indices: [n_tokens, topk]

        Returns:
            output: [n_tokens, hidden_dim]
        """
        results = torch.zeros_like(hidden_flat)

        for i, expert in enumerate(self.experts):
            # Find which (token, k) pairs selected this expert
            token_idx, topk_idx = torch.where(topk_indices == i)

            if len(token_idx) == 0:
                continue  # No tokens routed to this expert

            # Get the tokens and their weights for this expert
            expert_input = hidden_flat[token_idx]  # [num_selected, hidden_dim]
            weights = topk_weights[token_idx, topk_idx].unsqueeze(
                -1
            )  # [num_selected, 1]

            # Run expert and accumulate weighted output
            expert_output = expert(expert_input)  # [num_selected, hidden_dim]
            results[token_idx] += weights * expert_output

        return results


# =============================================================================
# Loss Functions (reused from moe.py, but included here for convenience)
# =============================================================================


def compute_load_balance_loss(
    router_probs: Tensor, topk_indices: Tensor, n_experts: int
) -> Tensor:
    """
    Auxiliary load balancing loss (Switch Transformer / Mixtral style).

    Encourages balanced expert usage by combining two signals:
    - f_i: fraction of total routing assignments to expert i (for top-k routing,
          each token contributes k assignments, so sum of f_i = 1.0)
    - P_i: mean PROBABILITY assigned to expert i (soft signal)

    Loss = n_experts * Σ(f_i * P_i)

    Minimum value is 1.0 when perfectly balanced.

    Args:
        router_probs: [n_tokens, n_experts] - softmax probabilities from router
        topk_indices: [n_tokens, topk] - selected expert indices
        n_experts: Total number of experts

    Returns:
        Scalar loss value
    """
    n_tokens = router_probs.shape[0]
    topk = topk_indices.shape[1]

    # P_i: mean probability assigned to each expert
    mean_probs = router_probs.mean(dim=0)  # [n_experts]

    # f_i: fraction of tokens routed to each expert
    flat_indices = topk_indices.view(-1)
    expert_counts = torch.bincount(flat_indices, minlength=n_experts).float()
    total_assignments = n_tokens * topk
    token_fractions = expert_counts / total_assignments  # [n_experts]

    # Load balancing loss
    return n_experts * (token_fractions * mean_probs).sum()


def compute_z_loss(router_logits: Tensor) -> Tensor:
    """
    Z-loss for router logit stabilization (ST-MoE, Zoph et al. 2022).

    Penalizes large logit magnitudes to prevent:
    - Extremely peaked softmax (kills exploration)
    - Numerical instability (NaN/Inf)
    - Dead experts with very negative logits

    Args:
        router_logits: [n_tokens, n_experts] - raw logits before softmax

    Returns:
        Scalar loss value
    """
    logsumexp = torch.logsumexp(router_logits, dim=-1)  # [n_tokens]
    return torch.mean(logsumexp**2)


# =============================================================================
# GPT-2 Surgery: Installing MoE Layers
# =============================================================================


def install_moe_layers(
    model,  # GPT2LMHeadModel
    moe_layers: list[int] = [8, 9, 10, 11],
    n_experts: int = 8,
    topk: int = 1,
    noise_std: float = 0.1,
) -> tuple:
    """
    Replace specified GPT-2 MLP layers with MoE wrappers.

    This is "model surgery" — we swap out the MLP modules but leave
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
        # Get the transformer block
        block = model.transformer.h[layer_idx]

        # Get the original MLP
        original_mlp = block.mlp

        # Create MoE wrapper (experts initialized as copies of original)
        moe = MoEWrapper(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
        )

        # Replace the MLP with MoE wrapper
        block.mlp = moe

        # Store reference for auxiliary output retrieval
        moe_modules[layer_idx] = moe

        print(
            f"Installed MoE at layer {layer_idx}: "
            f"{n_experts} experts, top-{topk} routing"
        )

    return model, moe_modules


def collect_aux_outputs(moe_modules: dict) -> list[dict]:
    """
    Collect auxiliary outputs from all MoE layers after a forward pass.

    Call this after model(input_ids) to get routing info for loss computation.

    Args:
        moe_modules: Dict from install_moe_layers()

    Returns:
        List of dicts with routing stats per layer (probs, logits, indices, entropy)
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


# =============================================================================
# Quick Verification
# =============================================================================

if __name__ == "__main__":
    # Quick test without actually loading GPT-2 (to avoid large download)
    print("Testing MoEWrapper with a dummy MLP...")

    # Create a dummy MLP similar to GPT2MLP
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

    # Test forward pass
    x = torch.randn(2, 10, 768)  # [batch=2, seq=10, hidden=768]
    output = moe(x)

    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Shapes match: {x.shape == output.shape}")

    # Check aux outputs
    aux = moe.last_aux
    print(f"✓ Router probs shape: {aux.router_probs.shape}")  # [20, 8]
    print(f"✓ Topk indices shape: {aux.topk_indices.shape}")  # [20, 1]

    # Test loss computation
    lb_loss = compute_load_balance_loss(aux.router_probs, aux.topk_indices, n_experts=8)
    z_loss = compute_z_loss(aux.router_logits)
    print(f"✓ Load balance loss: {lb_loss.item():.4f} (target ~1.0)")
    print(f"✓ Z-loss: {z_loss.item():.4f}")

    print("\n✅ MoEWrapper ready for GPT-2 integration!")
