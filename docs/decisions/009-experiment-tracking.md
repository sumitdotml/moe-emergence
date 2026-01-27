# Decision: Experiment Tracking with W&B + Modal

**Date:** 2025-01-28
**Status:** Accepted
**Context Commit:** `c596a05`

---

## Context

Need to track training metrics for MoE experiments, visualize expert specialization emergence, and produce publication-quality figures for the technical report. The tracking solution must work with Modal (cloud GPU provider) and support the specific metrics needed for MoE analysis.

---

## Options Considered

### Option A: Weights & Biases (W&B)

**Description:** Cloud-hosted experiment tracking with rich visualization, Modal integration, and team collaboration features.

**Pros:**
- Native Modal integration (documented in Modal examples)
- Rich visualization: heatmaps, custom charts, comparison tables
- Automatic hyperparameter logging
- Easy to share results and embed figures in reports
- Free tier sufficient for this project

**Cons:**
- Requires account and API key
- Data stored externally (cloud)

### Option B: TensorBoard

**Description:** Google's visualization toolkit, commonly used with PyTorch.

**Pros:**
- No account required
- Local data storage
- Built into PyTorch ecosystem

**Cons:**
- Visualization less flexible for custom metrics
- No native Modal integration
- Harder to share results
- Heatmaps require custom code

### Option C: MLflow

**Description:** Open-source platform for ML lifecycle management.

**Pros:**
- Open source, self-hostable
- Good experiment comparison

**Cons:**
- Heavier setup for Modal
- Overkill for this project's scope
- Less polished visualizations

### Option D: Simple CSV Logs

**Description:** Log metrics to CSV files, generate plots with matplotlib.

**Pros:**
- No external dependencies
- Full control over data

**Cons:**
- Manual work for every visualization
- No real-time monitoring
- No experiment comparison UI
- Error-prone

---

## Decision

**Option A: W&B + Modal integration**

W&B provides the best balance of features, ease of use, and Modal compatibility. The free tier is sufficient for this project's ~4 runs, and the visualization capabilities directly support the figures needed for the technical report.

### What to Log

**Training Metrics (per step):**
- Total loss
- Per-domain loss (code, math, prose)
- Load balancing loss
- Z-loss (router stabilization)

**Router Metrics (per step or periodic):**
- Per-expert utilization (fraction of tokens routed to each expert)
- Router entropy (measures routing diversity)
- Expert-domain affinity matrix (8 experts × 3 domains)

**Evaluation Metrics (per eval):**
- Per-domain eval loss
- Per-domain perplexity

### Visualizations for Paper

1. **Loss curves:** MoE vs dense baseline (line plot)
2. **Expert specialization heatmap:** 8 experts × 3 domains (heatmap)
3. **Ablation comparison:** With vs without load balancing (grouped bar chart)
4. **Per-domain eval loss:** Final performance by domain (bar chart)
5. **Router entropy over training:** Convergence to specialization (line plot)

---

## Consequences

**Positive:**
- Real-time monitoring during training
- Easy comparison between runs (dense vs MoE, ablations)
- Publication-ready figures exportable from W&B
- Reproducibility: all hyperparameters logged automatically

**Negative:**
- Requires W&B account setup
- API key management in Modal secrets

**Risks:**
- W&B service disruption during training (mitigated: also log locally)

---

## References

- [Modal + W&B integration docs](https://modal.com/docs/examples/wandb_logging)
- [W&B PyTorch integration](https://docs.wandb.ai/guides/integrations/pytorch)
- `docs/project-design/MOE-PROJECT-DESIGN-V3.md` (visualization requirements)
