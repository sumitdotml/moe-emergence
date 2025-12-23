# ST-MoE: Designing Stable and Transferable Sparse Expert Models

**Authors:** Zoph, Bello, et al.
**Year:** 2022
**Venue:** arXiv (Google Research)
**Link:** https://arxiv.org/abs/2202.08906

---

## TL;DR

Identifies training instability issues in MoE models and introduces z-loss to stabilize router logits. Also studies transfer learning with MoE.

---

## Key Claims

1. Router logit explosion is a major source of MoE training instability
2. Z-loss (penalizing log-sum-exp of logits) stabilizes training
3. Careful auxiliary loss balancing is crucial
4. MoE models transfer well when properly stabilized

---

## Relevant to This Project

Primary reference for:
- Z-loss formulation
- Understanding router instability failure mode
- Auxiliary loss coefficient tuning

---

## Key Equations

### Z-Loss (Router Logit Stabilization)

```
L_z = β * mean( logsumexp(router_logits, dim=-1)² )

where:
  β = coefficient (typically 1e-3 to 1e-2)
```

**Used in:** `moe.py:462-518`

**Intuition:**
- logsumexp ≈ max(logits) when one logit dominates
- Penalizing logsumexp² discourages extreme logit values
- Keeps softmax from becoming too peaked
- Prevents "dead experts" (very negative logits)

### Why logsumexp?

```
logsumexp([2, 1, 0, -1]) ≈ 2.4   # Reasonable
logsumexp([50, -30, -25, -40]) ≈ 50  # One expert dominates → heavily penalized
```

The squared penalty makes extreme values hurt quadratically.

---

## Architecture Details

- Studies both top-1 and top-2 routing
- Recommends z-loss coefficient 1e-3 to 1e-2
- Notes that z-loss is more important than load balancing for stability

---

## Hyperparameters Mentioned

| Parameter | Value | Notes |
|-----------|-------|-------|
| Z-loss coef | 1e-3 to 1e-2 | We use 0.001 |
| Load balance coef | 1e-2 | Combined with z-loss |

---

## Quotes

> "We find that the router z-loss, which penalizes large logits entering the router, is critical to prevent training instability."

(Abstract)

> "Without the router z-loss, we observed many runs diverge or produce NaN values."

(Section on training stability)

---

## Failure Mode Described

Without z-loss:
1. Router logits drift to extreme values
2. Softmax becomes extremely peaked (one expert → 0.99, others → 0.001)
3. Only one expert gets gradients
4. Other experts become "dead" (very negative logits, never selected)
5. Load balancing tries to fix this but can't overcome extreme logits
6. Training becomes unstable, eventually NaN

Z-loss prevents step 1, so cascade never starts.

---

## Questions / Critiques

- Paper is at large scale (billions of params). Effect at small scale may differ.
- Optimal z-loss coefficient may need tuning for our setup.

---

## Follow-up Papers

- Mixtral (2023) - doesn't explicitly mention z-loss but likely uses similar techniques
- GLaM (2022) - another large MoE study
