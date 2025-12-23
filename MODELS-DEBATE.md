---
Session Summary: 2025-12-23
Participants: GPT-5.2
Commit Reviewed: 31252b6ddc4a73c8776ffdf8ca53fc79988f35c2
---

Reviewed moe-emergence/moe.py against README.md and project-design/MOE-PROJECT-DESIGN-V3.md; below are the
critical findings focused on load balancing and MoE behavior (ignoring the SwiGLU expert choice as requested).

Findings

- Medium: Router noise is effectively disabled unless set_noise_annealing is called (anneal_steps starts at 0),
  but the docstring implies noise_std is active by default; this can silently remove symmetry breaking and
  diverges from the V3 plan. moe-emergence/moe.py:94 moe-emergence/moe.py:172
- Medium: Router only returns noisy router_probs; there is no clean (pre-noise) probability output, so entropy/
  specialization analysis will be confounded by annealing, contrary to V3’s “clean vs routing” separation. moe-
  emergence/moe.py:186 moe-emergence/moe.py:261
- Low: Router assumes 3D input [batch, seq, hidden]; if you follow the V3 pseudocode or reuse it with flattened
  [tokens, hidden], it will error. Consider supporting 2D or making the contract explicit. moe-emergence/
  moe.py:130
- Low: .view on potentially non-contiguous tensors can throw at runtime; .reshape is safer for x and x_flat. moe-
  emergence/moe.py:133 moe-emergence/moe.py:374
- Low: Load-balance docstring says “fraction of tokens routed,” but for top‑k>1 you compute fraction of
  assignments (sum to 1). Functionally fine, but the wording is misleading. moe-emergence/moe.py:400 moe-
  emergence/moe.py:451

Open Questions

- Do you want noisy routing enabled by default when noise_std > 0, or do you prefer the explicit
  set_noise_annealing gate?
- Should Router accept 2D flattened inputs to align with the V3 wrapper plan and reduce integration risk?
- Do you want router_probs_clean (and possibly entropy) added to RouterOutput for analysis/logging?

---

Session Summary: 2025-12-23

Participants: Claude Opus 4.5 (this session), GPT-5.2 (prior review)
Commit Reviewed: 31252b6ddc4a73c8776ffdf8ca53fc79988f35c2

---

What Was Done

1. Independent Code Review of moe.py

I conducted a thorough review against the V3 design spec and compared findings with GPT-5.2's earlier review. Both reviewers converged on the same issues:

| ID  | Severity | Issue                                                                                          |
| --- | -------- | ---------------------------------------------------------------------------------------------- |
| H1  | HIGH     | Missing router_probs_clean and entropy in RouterOutput - will confound specialization analysis |
| M1  | MEDIUM   | Noise disabled by default (anneal_steps=0) - docstring misleading                              |
| L1  | LOW      | .view() should be .reshape() for non-contiguous tensors                                        |
| L2  | LOW      | Router assumes 3D input, no 2D support                                                         |
| L3  | LOW      | Load balancing docstring wording imprecise for top-k > 1                                       |

Verified as Correct: STE implementation, load balancing loss formula, z-loss, using noisy probs for load balancing.

---

2. Created Documentation Infrastructure

docs/
├── README.md # Index
├── WORKFLOW.md # Quick reference
├── code-reviews/
│ └── 001-2025-12-23-moe-py-review.md # Full review with findings
├── decisions/
│ ├── 001-base-model-choice.md # Why GPT-2
│ ├── 002-moe-last-4-layers.md # Why layers 8-11
│ └── 003-warm-start-via-deepcopy.md # Why deepcopy
├── experiments/
│ └── \_TEMPLATE.md
└── literature/
├── fedus-2021-switch-transformer.md # Load balancing source
└── zoph-2022-st-moe.md # Z-loss source

---

3. Created Slash Commands for Frictionless Documentation

/doc-decision <choice> # Log design decision
/doc-experiment <run> # Log training run
/doc-review <file> # Code review against V3
/doc-paper <paper> # Literature notes
/doc-fix <bug> # Document bug fix
/doc-status # Project status summary

---

4. Updated AGENTS.md

Added:

- Section 0: Documentation workflow with command table
- Section 3: Project context, architecture decisions, known issues, budget constraints

Any new model instance now has full context on the documentation workflow and project state.

---

Next Steps (Before Training)

1. Fix H1: Add router_probs_clean and entropy to RouterOutput
2. Fix M1: Either enable noise by default OR fix docstring
3. Fix L1: Replace .view() with .reshape()
4. Commit fixes, then run /doc-review moe-emergence/moe.py to verify

---

Key Files to Read

- project-design/MOE-PROJECT-DESIGN-V3.md - Authoritative spec
- docs/code-reviews/001-2025-12-23-moe-py-review.md - Full review findings
- AGENTS.md - Workflow instructions and project context

---
