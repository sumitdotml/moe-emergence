import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, NamedTuple
from ffn import SwiGLU as Expert


class RouterOutput(NamedTuple):
    topk_weights: Tensor  # [n_tokens, topk] - renormalized weights for selected experts
    topk_indices: Tensor  # [n_tokens, topk] - which experts were selected
    router_probs: (
        Tensor  # [n_tokens, n_experts] - full probability distribution (for load balancing)
    )


class Router(nn.Module):
    r"""
    Mixture of Experts Router. Routes each token to the top-k most suitable experts
    based on learned routing weights, enabling sparse activation in MoE architectures.

    On a high level, the router creates a learned specialization where different experts become
    proficient at different types of tokens (e.g., physics vs. geography content),
    enabling efficient sparse computation while maintaining model capacity.

    The router performs the following operations:
    ```
    scores = W_router · x                    (compute affinity for each expert)
    probabilities = softmax(scores)          (normalize across all experts)
    top_k_probs, top_k_indices = topk(probabilities, k)  (select best k experts)
    weights = renormalize(top_k_probs)       (ensure selected weights sum to 1)
    ```

    Process:
    1. **Scoring**: Projects input tokens to expert scores via learned linear layer
    2. **Softmax**: Converts scores to probabilities (ensures all 8 sum to 1.0)
    3. **Top-k Selection**: Picks k experts with highest probabilities per token
    4. **Renormalization**: Rescales selected weights to sum to 1.0

    Args:
        hidden_dim: Dimension of input token representations (e.g., 4096 in Mixtral)
        n_ffn_experts: Total number of available experts (e.g., 8 in Mixtral)
        topk: Number of experts to activate per token (e.g., 2 in Mixtral)
        device: Device to place router parameters on
        dtype: Data type for router parameters

    Shapes:
        Input: [batch_size, seq_len, hidden_dim]
        Output weights: [batch_size * seq_len, topk]
        Output indices: [batch_size * seq_len, topk]

    Returns:
        tuple: (weights, indices) where:
            - weights: Normalized routing weights for selected experts [total_tokens, topk]
            - indices: Expert indices selected for each token [total_tokens, topk]

    Example:
        >>> router = MoERouter(hidden_dim=4096, n_ffn_experts=8, topk=2)
        >>> x = torch.randn(2, 10, 4096)  # 2 sequences, 10 tokens each
        >>> weights, indices = router(x)
        >>> weights.shape  # [20, 2] - each of 20 tokens routes to 2 experts
        >>> indices.shape  # [20, 2]
        >>> weights.sum(dim=-1)  # All close to 1.0 (renormalized)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_ffn_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.topk = topk
        self.W_router = nn.Linear(
            in_features=hidden_dim,
            out_features=n_ffn_experts,
            bias=False,
            **factory_kwargs,
        )

    # x is [batch, seqlen, hidden_dim]
    def forward(self, x: Tensor) -> RouterOutput:
        """
        being a bit comprehensive & verbose here as I'm learning it all, might clean this
        up later by for instance, getting rid of unused names & comments. but for now it's chill
        """
        # Step 1: flattening batch and sequence dimensions
        _, _, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch*seqlen, hidden_dim]

        # Step 2: raw scores' computation for all experts
        router_matrix = self.W_router(x_flat)  # [batch*seqlen, n_ffn_experts]

        # Step 3: softmax, gives full probability distribution over all experts for load balancing loss computation
        router_probs = torch.softmax(router_matrix, dim=1)  # [n_tokens, n_experts]

        # Step 4: top-k experts' selection
        # topk_weights: [batch*seqlen, topk], topk_indices: [batch*seqlen, topk]
        topk_weights, topk_indices = torch.topk(router_probs, k=self.topk)

        # Step 5: weights' renormalization
        topk_weights = self.renormalization(topk_weights)

        return RouterOutput(topk_weights, topk_indices, router_probs)

    def renormalization(self, input: Tensor) -> Tensor:
        total = input.sum(
            dim=-1, keepdim=True
        )  # total sum of experts' raw scores, not token count, hence -1 dim
        renormalized = input / (total + 1e-9)  # [batch*seqlen, topk]
        return renormalized


class MoEOutput(NamedTuple):
    hidden_states: Tensor
    balance_loss: Tensor


class MoE(nn.Module):
    r"""
    Mixture of Experts Layer. Replaces a standard dense feed-forward network with
    multiple expert FFNs, routing each token to its top-k most suitable experts for
    sparse, efficient computation while maintaining model capacity.

    Architecture:
    ```
    Input [batch, seq, hidden_dim]
        ↓
    Router: determines which k experts handle each token
        ↓
    Expert FFNs: selected experts process their assigned tokens in parallel
        ↓
    Weighted Combination: expert outputs combined using router weights
        ↓
    Output [batch, seq, hidden_dim]
    ```

    Mathematical formulation:
    ```
    For each token x:
        topk_weights, topk_indices = Router(x)              # Select top-k experts
        output = Σ(weights[i] * Expert[i](x))     # Weighted combination
    ```

    Computational Efficiency:
    - Dense FFN: Uses all parameters for every token
    - MoE: Only k out of n_experts are active per token
    - Example: 8 experts, top-2 → 75% parameter reduction per token

    Args:
        hidden_dim: Dimension of token representations (e.g., 4096 in Mixtral)
        ffn_dim: Intermediate dimension for expert FFNs (typically 3-4× hidden_dim)
        n_experts: Total number of expert FFNs (e.g., 8 in Mixtral 8x7B)
        topk: Number of experts activated per token (e.g., 2 in Mixtral)
        device: Device to place parameters on
        dtype: Data type for parameters

    Shapes:
        Input: [batch_size, seq_len, hidden_dim]
        Output: [batch_size, seq_len, hidden_dim]

    Example:
        >>> # Mixtral-style configuration
        >>> moe = MoE(
        ...     hidden_dim=4096,
        ...     ffn_dim=14336,
        ...     n_experts=8,
        ...     topk=2
        ... )
        >>> x = torch.randn(2, 512, 4096)  # 2 sequences, 512 tokens
        >>> output = moe(x)
        >>> output.shape  # [2, 512, 4096]

    Note:
        During training, experts naturally specialize in different types of content
        (e.g., code vs prose, technical vs conversational). This emergent specialization
        enables efficient sparse computation without sacrificing the model's ability to
        handle diverse inputs.

        The layer processes tokens by batching all tokens assigned to each expert,
        running each expert once per forward pass for maximum efficiency.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        n_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk
        self.router = Router(
            hidden_dim=hidden_dim, n_ffn_experts=n_experts, topk=topk, **factory_kwargs
        )
        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim=hidden_dim, ffn_dim=ffn_dim, **factory_kwargs)
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: Tensor) -> MoEOutput:
        batch, seqlen, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # router does flattening internally, so no need to pass x_flat; just x is fine
        topk_weights, topk_indices, router_probs = self.router(x)
        results = torch.zeros_like(x_flat)

        # reference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py#L16
        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(topk_indices == i)
            if len(token_idx) == 0:
                continue
            # unsqueeze is to add 1 dim at the end since weights[token_idx, topk_idx] gives us 1d-tensors
            results[token_idx] += topk_weights[token_idx, topk_idx].unsqueeze(-1) * expert(
                x_flat[token_idx]
            )

        balance_loss = self.compute_load_balance_loss(router_probs, topk_indices)
        return MoEOutput(results.view(batch, seqlen, hidden_dim), balance_loss)

    def compute_load_balance_loss(self, router_probs: Tensor, topk_indices: Tensor) -> Tensor:
        """
        Computes auxiliary load balancing loss (Switch Transformer / Mixtral style).

        The loss encourages balanced expert usage by combining two signals:
        - f_i: fraction of tokens ROUTED to expert i (hard assignment from top-k)
        - P_i: mean PROBABILITY assigned to expert i (soft signal from router)

        Loss = n_experts * Σ(f_i * P_i)

        Mental model (for myself):
        - If expert 1 is overloaded (high f_1), the loss penalizes high P_1
        - Gradient flows through P_i (differentiable) to reduce routing probability
        - This eventually reduces f_1 as fewer tokens get routed there

        Theoretical minimum: 1.0 (when all experts are perfectly balanced)
        - Perfect balance: f_i = P_i = 1/n_experts for all experts
        - Loss = n_experts * n_experts * (1/n_experts)² = 1.0

        Args:
            router_probs: [n_tokens, n_experts] - full probability distribution from router
            topk_indices: [n_tokens, topk] - which experts were selected for each token

        Returns:
            Scalar auxiliary loss to be added to main loss

        Example (during training):
            >>> moe = MoE(hidden_dim=4096, ffn_dim=14336, n_experts=8, topk=2)
            >>> output, balance_loss = moe(x)
            >>> total_loss = main_loss + α * balance_loss  # α typically 0.01 - 0.1
        """
        n_tokens = router_probs.shape[0]

        # P_i: mean probability assigned to each expert across all tokens
        # this is differentiable and provides gradient signal
        # shape: [n_experts]
        mean_probs = router_probs.mean(dim=0)

        # f_i: fraction of tokens routed to each expert
        # counts how many times each expert appears in top-k selections
        # one_hot: [n_tokens, topk, n_experts] -> sums over tokens & topk positions
        expert_counts = torch.zeros(
            self.n_experts, device=router_probs.device, dtype=router_probs.dtype
        )
        for i in range(self.n_experts):
            expert_counts[i] = (topk_indices == i).sum()

        # normalizing by total assignments to get fractions: each token picks topk experts, so
        # total assignments = n_tokens * topk
        total_assignments = n_tokens * self.topk
        token_fractions = expert_counts / total_assignments  # [n_experts]

        # load balancing loss: n_experts * Σ(f_i * P_i)
        # the n_experts scaling to ensure minimum loss of 1.0 at perfect balance
        aux_loss = self.n_experts * (token_fractions * mean_probs).sum()

        return aux_loss
