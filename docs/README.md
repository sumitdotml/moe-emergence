# Documentation Index

This directory contains all project documentation for the MoE Emergence study. Maintained for:

1. Personal learning record
2. Technical report writing
3. Reproducibility

---

## Directory Structure

```
docs/
├── README.md                 # This file
├── code-reviews/             # Code review logs with findings
├── decisions/                # Decision log (why choices were made)
├── experiments/              # Experiment logs (runs, hyperparams, results)
└── literature/               # Paper notes and references
```

---

## Documentation Types

### 1. Code Reviews (`code-reviews/`)

Formal reviews of implementation against spec. Each review documents:

- Commit hash reviewed
- Findings by severity
- Comparison with design docs
- Next steps

**Naming:** `NNN-YYYY-MM-DD-{component}-review.md`

---

### 2. Decision Log (`decisions/`)

Records **why** design choices were made, not just what. Critical for:

- Defending choices in technical report
- Remembering rationale months later
- Explaining tradeoffs to reviewers

**Template:** See `decisions/_TEMPLATE.md`

---

### 3. Experiment Log (`experiments/`)

Every training run gets logged:

- Hyperparameters (exact config)
- Hardware/environment
- Results (loss curves, metrics)
- Observations and anomalies
- Cost (GPU hours, $)

**Naming:** `run-{NNN}-{short-description}.md`

---

### 4. Literature Notes (`literature/`)

Notes on papers referenced:

- Key claims
- Equations used in implementation
- How it relates to this project

**Naming:** `{author}-{year}-{short-title}.md`

---

## What to Document When

| Event                           | Document Type                 | Priority |
| ------------------------------- | ----------------------------- | -------- |
| Design choice made              | Decision Log                  | HIGH     |
| Code reviewed                   | Code Review                   | HIGH     |
| Training run completed          | Experiment Log                | HIGH     |
| Bug found and fixed             | Decision Log (or code review) | MEDIUM   |
| Paper read                      | Literature Notes              | MEDIUM   |
| Implementation approach changed | Decision Log                  | HIGH     |

---

## For the Technical Report

When writing the final report, pull from:

1. **Methodology section:** Decision log entries
2. **Results section:** Experiment logs
3. **Related work:** Literature notes
4. **Appendix:** Code review findings (shows rigor)

---

## Quick Links

- [Code Reviews](./code-reviews/)
- [Experiments](./experiments/)
- [V3 Design Spec](./project-design/MOE-PROJECT-DESIGN-V3.md)
- [Project README](../README.md)
- [Agent Instructions](../AGENTS.md)

## Experiment Logs

- `docs/experiments/run-001-gpt2-integration-verification.md` (Phase 2 verification)

---

## Code Reviews Summary

| #   | Date       | Component           | Status             |
| --- | ---------- | ------------------- | ------------------ |
| 001 | 2025-12-23 | `moe.py` review     | Fixed in `eea9294` |
| 002 | 2025-12-23 | `moe.py` fix report | —                  |
| 003 | 2025-12-23 | `gpt2_moe.py` gaps  | Fixed in `ffc77ab` |
| 004 | 2025-12-23 | Loss dedup + tests  | Fixed in `c929d8c` |

---

## Project Status

**Current Phase:** Phase 2 COMPLETE (GPT-2 integration verified)

**Last Verification:** run-001 (2025-12-25) - All 10 tests passed (merry christmas!)

**Next Steps:**

1. ~~Verify GPT-2 integration with actual HuggingFace model~~ ✅ DONE
2. Phase 3: Dataset preparation (sequence packing, data collection)
3. Phase 4: Training infrastructure
