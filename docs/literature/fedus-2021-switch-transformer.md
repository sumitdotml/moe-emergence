# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**Authors:** Fedus, Zoph, Shazeer
**Year:** 2021
**Venue:** JMLR
**Link:** https://arxiv.org/abs/2101.03961

---

## TL;DR

Simplifies MoE by using top-1 routing (each token → 1 expert), introduces load balancing auxiliary loss, scales to trillion parameters.

---

## Key Claims

1. Top-1 routing works well (simpler than top-k, reduces communication in distributed setting)
2. Load balancing loss is necessary to prevent expert collapse
3. MoE can scale to 1.6T parameters while keeping compute manageable
4. Sparse models can match dense models at lower compute cost

---

## Relevant to This Project

This is the primary reference for:
- Load balancing loss formulation
- Top-1 routing design
- Expert collapse problem definition

---

## Key Equations

### Load Balancing Loss

```
L_balance = α * N * Σᵢ fᵢ * Pᵢ

where:
  N = number of experts
  fᵢ = fraction of tokens routed to expert i (hard assignment)
  Pᵢ = mean probability assigned to expert i (soft, differentiable)
  α = coefficient (typically 0.01)
```

**Used in:** `moe.py:394-460`

**Intuition:**
- If expert i is overloaded (high fᵢ), loss penalizes high Pᵢ
- Gradient flows through Pᵢ (differentiable) to reduce routing to overloaded experts
- Minimum loss = 1.0 when all experts equally loaded (fᵢ = Pᵢ = 1/N)

### Why N * Σ(f*P)?

At perfect balance:
- fᵢ = 1/N for all i
- Pᵢ = 1/N for all i
- Loss = N * N * (1/N)² = 1.0

The N scaling ensures minimum is exactly 1.0, making it easy to interpret.

---

## Architecture Details

- Uses top-1 routing (each token to exactly 1 expert)
- Capacity factor: limits max tokens per expert (we don't use this)
- Token dropping: drops tokens when expert at capacity (we don't use this)

---

## Hyperparameters Mentioned

| Parameter | Value | Notes |
|-----------|-------|-------|
| Load balance coef | 0.01 | They use α = 0.01 |
| Capacity factor | 1.0-2.0 | We don't use capacity limiting |
| Experts | 128-2048 | We use 8 |

---

## Quotes

> "We find that the simple strategy of routing to a single expert works well in practice"

(Section 2.1)

> "Without this auxiliary loss, we observe that the weights quickly converge to a state where the weights for all experts are equal, which fails to leverage the additional capacity of the experts."

(Section 2.2, on why load balancing is necessary)

---

## Questions / Critiques

- Paper focuses on distributed training at massive scale. Our single-GPU setup may have different dynamics.
- Capacity factor and token dropping are important at scale but we skip them. May affect collapse behavior.

---

## Follow-up Papers

- ST-MoE (Zoph et al., 2022) - adds z-loss for stability
- Mixtral (2023) - production MoE with top-2 routing
- GShard (Lepikhin et al., 2020) - earlier large-scale MoE
