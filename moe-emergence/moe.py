import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, NamedTuple
from ffn import SwiGLU as Expert


class RouterOutput(NamedTuple):
    topk_weights: Tensor  # [n_tokens, topk] - renormalized weights for selected experts
    topk_indices: Tensor  # [n_tokens, topk] - which experts were selected per token
    router_probs: Tensor  # [n_tokens, n_experts] - probs for routing (post-noise, for load balancing)
    router_probs_clean: (
        Tensor  # [n_tokens, n_experts] - clean probs before noise (for entropy logging)
    )
    router_logits: (
        Tensor  # [n_tokens, n_experts] - raw logits before softmax (for z-loss)
    )
    entropy: Tensor  # [n_tokens] - router entropy computed from clean probs


class Router(nn.Module):
    r"""
    Mixture of Experts Router (Gating Network).

    Routes each token to the top-k most suitable experts based on learned routing
    weights, enabling sparse activation in MoE architectures.

    On a high level, the router creates a learned specialization where different experts become
    proficient at different types of tokens (e.g., physics vs. geography content),
    enabling efficient sparse computation while maintaining model capacity.

    The router performs the following operations:
    ```
    scores = gate(x)                         (compute affinity for each expert)
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
        n_experts: Total number of available experts (e.g., 8 in Mixtral)
        topk: Number of experts to activate per token (e.g., 2 in Mixtral)
        noise_std: Standard deviation for NoisyTop-k routing. Noise is only active
            after calling set_noise_annealing(). (default 0.1)
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
        >>> router = Router(hidden_dim=4096, n_experts=8, topk=2)
        >>> x = torch.randn(2, 10, 4096)  # 2 sequences, 10 tokens each
        >>> out = router(x)  # RouterOutput with 6 fields
        >>> out.topk_weights.shape  # [20, 2]
        >>> out.topk_indices.shape  # [20, 2]
        >>> out.router_probs_clean.shape  # [20, 8] - for entropy logging
        >>> out.entropy.shape  # [20] - per-token routing entropy

    References:
        - Noisy Top-k Gating: Shazeer et al. "Outrageously Large Neural Networks" (2017)
        - STE for Top-1: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
    """

    training_step: Tensor
    anneal_steps: Tensor

    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        topk: int,
        noise_std: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert topk <= n_experts, f"topk ({topk}) cannot exceed n_experts ({n_experts})"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.topk = topk
        self.noise_std = noise_std
        self.gate = nn.Linear(
            in_features=hidden_dim,
            out_features=n_experts,
            bias=False,
            **factory_kwargs,
        )

        ###################################################
        ####### NOISY ROUTING STATE (for training) ########
        ###################################################
        # These buffers track training progress for noise annealing.
        # register_buffer: saved with model but not a learnable parameter
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("anneal_steps", torch.tensor(0, dtype=torch.long))

    def set_noise_annealing(self, total_steps: int, anneal_fraction: float = 0.25):
        """
        Configures the noise annealing schedule.
        This is used to gradually reduce the noise in the router logits over the course of training.

        Args:
            total_steps: Total training steps
            anneal_fraction: Fraction of training to anneal noise over (default 25%)

        Example:
            >>> router.set_noise_annealing(total_steps=10000, anneal_fraction=0.25)
            >>> # Noise will linearly decay to 0 over the first 2500 steps
        """
        self.anneal_steps.fill_(int(total_steps * anneal_fraction))

    def step(self):
        """To be called after each optimizer step to update noise schedule."""
        self.training_step += 1

    def forward(self, x: Tensor) -> RouterOutput:
        r"""
        Routes input tokens to their top-k experts.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            RouterOutput containing topk_weights, topk_indices, router_probs, router_logits
        """
        # x is [batch, seqlen, hidden_dim]
        # Step 1: flattening batch and sequence dimensions
        _, _, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [batch*seqlen, hidden_dim]

        # Step 2: raw logits computation for all experts
        router_logits = self.gate(x_flat)  # [batch*seqlen, n_experts]

        # Step 2.5: computation of CLEAN probabilities and entropy BEFORE adding noise
        # used for logging/analysis and should not be confounded by noise
        router_probs_clean = torch.softmax(
            router_logits, dim=1
        )  # [n_tokens, n_experts]
        entropy = -(router_probs_clean * torch.log(router_probs_clean + 1e-9)).sum(
            dim=-1
        )  # [n_tokens]

        #######################################################################
        ####### NOISY TOP-K ROUTING (NoisyTop-k from Shazeer et al.) ##########
        #######################################################################
        #
        # WHY ADD NOISE?
        # ──────────────
        # When experts start as identical copies (warm-start), the router has no
        # reason to prefer one over another. Without noise:
        #   • All experts get similar gradients
        #   • Specialization emerges slowly (or not at all)
        #   • Risk of "rich get richer" - one expert dominates by chance
        #
        # With noise:
        #   • Different tokens randomly explore different experts
        #   • Experts receive diverse gradients → faster divergence
        #   • More robust specialization emerges
        #
        # WHY ANNEAL (aka reduce) THE NOISE?
        # ───────────────────────────────────
        # Early training: high noise → exploration, experts diverge
        # Later training: low/no noise → exploitation, router's learned preferences
        #
        # If noise continues forever, reported "confident routing" might just be
        # the noise settling, not actual learned specialization.
        #
        # TIMELINE (for example):
        # ─────────────────────
        #   Step 0          Step 2500 (25%)        Step 10000
        #   |─────────────────|─────────────────────────|
        #   noise=0.1        noise=0.0                 noise=0.0
        #   (exploration)    (exploitation)
        #
        #######################################################################

        # to apply noise only during training, with linear annealing
        if (
            self.training
            and self.anneal_steps > 0
            and self.training_step < self.anneal_steps
        ):
            # Linear decay: noise_std * (1 - progress)
            progress = self.training_step.float() / self.anneal_steps.float()
            current_noise_std = self.noise_std * (1.0 - progress)
            noise = torch.randn_like(router_logits) * current_noise_std
            routing_logits = router_logits + noise
        else:
            routing_logits = router_logits

        # Step 3: softmax on (noisy if in annealing phase, otherwise not noisy) logits for routing
        router_probs = torch.softmax(routing_logits, dim=1)  # [n_tokens, n_experts]

        # Step 4: top-k experts' selection
        # topk_weights: [batch*seqlen, topk], topk_indices: [batch*seqlen, topk]
        topk_weights, topk_indices = torch.topk(router_probs, k=self.topk)

        # Step 5: handling weights based on top-k value
        if self.topk == 1:
            ##################################################################
            ####### STRAIGHT-THROUGH ESTIMATOR (STE) FOR TOP-1 ROUTING #######
            ##################################################################
            #
            # THE PROBLEM:
            # ────────────
            # For top-1, if we just use weight=1.0, the router never learns from
            # the main LM loss because 1.0 is a constant — it has no connection to
            # the router's parameters in the computation graph.
            #
            #   router.gate → logits → softmax → topk → ??? → 1.0 → expert → loss
            #                                               ↑
            #                                        DISCONNECTED!
            #                                  Gradient can't flow back
            #
            # THE SOLUTION (STE):
            # ───────────────────
            # We want: forward uses 1.0, but backward pretends we used soft probability.
            #
            # The trick: `hard + (soft - soft.detach())`
            #
            #   • soft.detach() = same VALUE as soft, but DETACHED from graph (no gradient)
            #   • soft - soft.detach() = 0 in forward (values cancel)
            #   • But in backward: d/d(soft)[soft - soft.detach()] = 1 - 0 = 1
            #     (because soft.detach() contributes zero gradient)
            #
            # FORWARD PASS:
            # ─────────────
            #   soft = 0.7, hard = 1.0
            #   result = 1.0 + (0.7 - 0.7) = 1.0  ← We get the hard value ✓
            #
            # BACKWARD PASS:
            # ──────────────
            #   Gradient flows through `soft` (not soft.detach())
            #   → softmax → logits → router.gate  ← Router learns! ✓
            #
            # COMPUTATION GRAPH (shoutout to Opus 4.5 for the beautiful diagram <3):
            # ──────────────────
            #
            #   router.gate ──→ logits ──→ softmax ──→ topk ──→ soft ────────┐
            #                                                        │       │
            #                                                        ↓       │
            #                                              soft.detach()     │
            #                                              (DEAD END -       │
            #                                               no gradient)     │
            #                                                                ↓
            #                       hard (1.0) ──────────────────────→ [+] ──→ weight ──→ expert ──→ loss
            #                                                          ↑
            #                                                   (soft - soft.detach() = 0,
            #                                                    but gradient flows through soft!)
            #
            # RESULT:
            # ───────
            #   • Forward: output = 1.0 (preserves scale, warm-start friendly)
            #   • Backward: gradient reaches router via soft (router learns from LM loss)
            #
            #######################################################################
            soft = topk_weights  # [batch*seq, 1] - the actual router probability
            hard = torch.ones_like(soft)  # [batch*seq, 1] - what we want in forward
            if self.training:
                topk_weights = hard + (soft - soft.detach())  # STE during training
            else:
                topk_weights = hard  # No STE overhead during inference
        else:
            # for top-k > 1: renormalizing so weights sum to 1
            # (relative weights still carry gradient info, no STE required)
            topk_weights = self._normalize_weights(topk_weights)

        # note to self: raw logits (before noise) are needed for z-loss computation
        return RouterOutput(
            topk_weights,
            topk_indices,
            router_probs,
            router_probs_clean,
            router_logits,
            entropy,
        )

    def _normalize_weights(self, weights: Tensor) -> Tensor:
        """Renormalizes top-k weights to sum to 1.0 per token."""
        total = weights.sum(
            dim=-1, keepdim=True
        )  # sum of experts' raw scores, not token count, hence dim=-1
        return weights / (total + 1e-9)  # [batch*seqlen, topk]


def compute_load_balance_loss(
    router_probs: Tensor, topk_indices: Tensor, n_experts: int
) -> Tensor:
    """
    Computes auxiliary load balancing loss (Switch Transformer / Mixtral style).

    The loss encourages balanced expert usage by combining two signals:
    - f_i: fraction of total routing assignments to expert i (for top-k routing,
          each token contributes k assignments, so sum of f_i = 1.0)
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
        n_experts: Total number of experts

    Returns:
        Scalar auxiliary loss to be added to main loss

    Example (during training):
        >>> moe = MoE(hidden_dim=4096, ffn_dim=14336, n_experts=8, topk=2)
        >>> output, balance_loss, z_loss = moe(x)
        >>> total_loss = lm_loss + α * balance_loss + β * z_loss  # α ~ 0.01, β ~ 0.001
    """
    n_tokens, _ = router_probs.shape

    # P_i: mean probability assigned to each expert across all tokens
    # this is differentiable and provides gradient signal
    # shape: [n_experts]
    mean_probs = router_probs.mean(dim=0)

    # f_i: fraction of tokens routed to each expert
    # counts how many times each expert appears in top-k selections
    # two ways to count: using loop or bincount (bincount is faster)

    # 1. using loop (I'll leave this here since this is easier to intuit for me)
    # expert_counts = torch.zeros(
    #     n_experts, device=router_probs.device, dtype=router_probs.dtype
    # )
    # for i in range(n_experts):
    #     expert_counts[i] = (topk_indices == i).sum()

    # 2. using bincount (faster than loop, not so intuitive for me; but I'll get used to it)
    flat_indices = topk_indices.view(-1)  # [n_tokens * topk]
    expert_counts = torch.bincount(flat_indices, minlength=n_experts).to(
        router_probs.dtype
    )  # [n_experts]

    # normalizing by total assignments to get fractions: each token picks topk experts, so
    # total assignments = n_tokens * topk
    total_assignments = n_tokens * topk_indices.shape[1]
    token_fractions = expert_counts / total_assignments  # [n_experts]

    # load balancing loss: n_experts * Σ(f_i * P_i)
    # the n_experts scaling to ensure minimum loss of 1.0 at perfect balance
    aux_loss = n_experts * (token_fractions * mean_probs).sum()

    return aux_loss


def compute_z_loss(router_logits: Tensor) -> Tensor:
    r"""
    Z-LOSS: Router Logit Stabilization (from ST-MoE, Zoph et al. 2022)

    THE PROBLEM:
    ────────────
    During training, router logits can drift to extreme values:

      • Very large positive logits → softmax becomes extremely peaked
        → one expert gets probability ≈1.0, others ≈0.0
        → kills exploration, router becomes overconfident

      • Very large negative logits → expert becomes effectively "dead"
        → even load balancing can't revive it easily

      • Numerical instability → NaN/Inf in softmax, training crashes

    Example of what can go wrong:
        logits = [50.0, -30.0, -25.0, -40.0, ...]
        probs  = [1.0,   0.0,   0.0,   0.0, ...]  # Expert 0 monopolizes

    THE SOLUTION (Z-LOSS):
    ──────────────────────
    Penalize large logit magnitudes via log-sum-exp:

        z_loss = mean( logsumexp(logits)² )

    WHY LOGSUMEXP?
    ──────────────
    logsumexp(x) ≈ max(x) when one logit dominates
    logsumexp(x) ≈ log(n) + mean(x) when logits are similar

    By penalizing logsumexp², we:
      • Discourage any single logit from being too large
      • Keep all logits in a reasonable range
      • Maintain numerical stability

    EXAMPLE:
    ───────

    Healthy logits:  [2.0, 1.5, 1.0, 0.5]  → logsumexp ≈ 2.8  → z_loss ≈ 7.8
    Unhealthy:       [50.0, -30, -25, -40] → logsumexp ≈ 50   → z_loss ≈ 2500

    The unhealthy case gets heavily penalized.

    USAGE (in training):
    ────────────────────
    total_loss = lm_loss + α * balance_loss + β * z_loss

    Typical β (z_loss coefficient): 1e-3 to 1e-2
    (Much smaller than balance_loss coefficient because z_loss values are larger)
    """
    # logsumexp along expert dimension
    logsumexp = torch.logsumexp(
        router_logits, dim=-1
    )  # shape: [n_tokens] - one value per token
    return torch.mean(logsumexp**2)


class MoEOutput(NamedTuple):
    hidden_states: Tensor
    balance_loss: Tensor
    z_loss: Tensor


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
        noise_std: Standard deviation of noise to add to logits (default 0.1)
        device: Device to place parameters on
        dtype: Data type for parameters

    Shapes:
        Input: [batch_size, seq_len, hidden_dim]
        Output: [batch_size, seq_len, hidden_dim]

    Example:
        >>> # Mixtral-style configuration
        >>> moe = MoE(hidden_dim=4096, ffn_dim=14336, n_experts=8, topk=2)
        >>> x = torch.randn(2, 512, 4096)  # 2 sequences, 512 tokens
        >>> output, balance_loss, z_loss = moe(x)
        >>> output.shape  # [2, 512, 4096]
        >>> balance_loss  # Scalar, target ~1.0
        >>> z_loss  # Scalar, for router stability. Prevents explosion of logits

    Note:
        During training, experts naturally specialize in different types of content
        (e.g., code vs prose, technical vs conversational). This emergent specialization
        enables efficient sparse computation without sacrificing the model's ability to
        handle diverse inputs.

        The layer processes tokens by batching all tokens assigned to each expert,
        running each expert once per forward pass for maximum efficiency.

    References:
        - Switch Transformer: Fedus et al. "Switch Transformers" (2021)
        - Mixtral: Jiang et al. "Mixtral of Experts" (2024)
        - ST-MoE (Z-loss): Zoph et al. "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        n_experts: int,
        topk: int,
        noise_std: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert topk <= n_experts, f"topk ({topk}) cannot exceed n_experts ({n_experts})"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk
        self.router = Router(
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            topk=topk,
            noise_std=noise_std,
            **factory_kwargs,
        )
        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim=hidden_dim, ffn_dim=ffn_dim, **factory_kwargs)
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: Tensor) -> MoEOutput:
        batch, seqlen, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        # router does flattening internally, so no need to pass x_flat; just x is fine
        (
            topk_weights,
            topk_indices,
            router_probs,
            router_probs_clean,
            router_logits,
            entropy,
        ) = self.router(x)
        # Note: router_probs_clean and entropy are available for logging but not used in forward pass
        results = torch.zeros_like(x_flat)

        # reference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/moe.py#L16
        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(topk_indices == i)
            if len(token_idx) == 0:
                continue
            # unsqueeze is to add 1 dim at the end since weights[token_idx, topk_idx] gives us 1d-tensors
            results[token_idx] += topk_weights[token_idx, topk_idx].unsqueeze(
                -1
            ) * expert(x_flat[token_idx])

        balance_loss = compute_load_balance_loss(
            router_probs, topk_indices, self.n_experts
        )
        z_loss = compute_z_loss(router_logits)
        return MoEOutput(results.view(batch, seqlen, hidden_dim), balance_loss, z_loss)
