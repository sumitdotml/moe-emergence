# Documentation Workflow Quick Reference

Print this or keep it open. Zero-friction documentation.

---

## Slash Commands

```
/doc-decision <what you decided>     # Log design choice
/doc-experiment <run description>    # Log training run
/doc-review <file path>              # Code review
/doc-paper <paper info>              # Literature notes
/doc-fix <what was fixed>            # Bug fix audit trail
/doc-status                          # Project status
```

---

## Daily Workflow

### Starting a Session
```
/doc-status
```
This shows what's done, what's pending, and suggests next action.

### After Making a Design Choice
```
/doc-decision chose X over Y because Z
```

### After Completing a Training Run
```
/doc-experiment run-004 main MoE training completed
```
Provide: config, results, observations, cost.

### After Fixing a Bug
```
/doc-fix fixed the router gradient flow issue
```

### Before Major Training Runs
```
/doc-review moe_emergence/moe.py
```
Ensures code matches V3 spec.

---

## File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Code review | `NNN-YYYY-MM-DD-{component}-review.md` | `001-2025-12-23-moe-py-review.md` |
| Decision | `NNN-{short-title}.md` | `003-warm-start-via-deepcopy.md` |
| Experiment | `run-NNN-{description}.md` | `run-001-dense-baseline.md` |
| Literature | `{author}-{year}-{title}.md` | `fedus-2021-switch-transformer.md` |

---

## What Gets Documented

| Event | Command | Creates |
|-------|---------|---------|
| Chose between options | `/doc-decision` | `docs/decisions/NNN-*.md` |
| Training run finished | `/doc-experiment` | `docs/experiments/run-NNN-*.md` |
| Code reviewed | `/doc-review` | `docs/code-reviews/NNN-YYYY-MM-DD-*.md` |
| Paper read | `/doc-paper` | `docs/literature/*.md` |
| Bug fixed | `/doc-fix` | Entry in code-review or new fix doc |

---

## For Technical Report

When writing the final report, pull from:

| Report Section | Source |
|----------------|--------|
| Methodology | `docs/decisions/` |
| Related Work | `docs/literature/` |
| Results | `docs/experiments/` |
| Appendix (rigor) | `docs/code-reviews/` |

---

## Golden Rule

**Every commit that changes implementation should have corresponding documentation.**

The documentation creates an audit trail that makes the technical report practically write itself.
