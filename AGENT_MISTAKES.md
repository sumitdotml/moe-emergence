# AGENT_MISTAKES

Persistent repository memory for recurring agent/model mistakes.

Initialized on 2026-02-17.

## Usage Rules

- Read this file before any repository edit task.
- Record every detected mistake occurrence.
- Deduplicate by normalized `pattern` + `scope_tags` + `prevention_rule`.
- For repeated patterns, update existing entry fields instead of creating duplicates.

## Required Entry Fields

Every entry must include:

- `id`
- `status` (`active` or `resolved`)
- `severity` (`low`, `medium`, or `high`)
- `scope_tags` (list)
- `pattern`
- `prevention_rule`
- `validation_check`
- `first_seen` (YYYY-MM-DD)
- `last_seen` (YYYY-MM-DD)
- `occurrence_count` (integer >= 1)
- `evidence` (one or more file:line and/or commit refs)

## Entry Template

Use this exact shape for new entries.

```md
### MISTAKE-YYYYMMDD-001
- id: MISTAKE-YYYYMMDD-001
- status: active
- severity: medium
- scope_tags: [code, docs, tests, config, infra, planning]
- pattern: <normalized mistake pattern>
- prevention_rule: <specific action that prevents recurrence>
- validation_check: <deterministic pass/fail check>
- first_seen: YYYY-MM-DD
- last_seen: YYYY-MM-DD
- occurrence_count: 1
- evidence:
  - file:relative/path:line
  - commit:<hash>
```

## Entries

### MISTAKE-20260225-001
- id: MISTAKE-20260225-001
- status: active
- severity: medium
- scope_tags: [docs]
- pattern: Using `--` (em-dash) for dramatic effect or list continuation instead of removable parenthetical phrases. Writing-style rule: em-dashes are ONLY for content that can be removed without losing core meaning.
- prevention_rule: Before using `--`, verify the clause between dashes can be deleted and the sentence still makes sense. If it can't, use a period, semicolon, or restructure.
- validation_check: For every `--` in edited text, mentally remove the dashed clause and confirm the sentence is grammatically complete and retains its core meaning.
- first_seen: 2026-02-25
- last_seen: 2026-02-25
- occurrence_count: 1
- evidence:
  - file:checkpoints/README.md:46 (attempted "Expert collapse at step 500 -- single expert handles 73.6%", rejected by user)

### MISTAKE-20260226-001
- id: MISTAKE-20260226-001
- status: active
- severity: high
- scope_tags: [code]
- pattern: catching broad Exception around optional third-party integration setup and continuing without re-raising non-integration errors.
- prevention_rule: when fallback behavior is needed, gate fallback by integration-origin checks and re-raise non-integration exceptions.
- validation_check: for integration init blocks, verify broad catches call an origin guard and re-raise when the guard fails.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:moe_emergence/train.py:525

### MISTAKE-20260226-002
- id: MISTAKE-20260226-002
- status: active
- severity: medium
- scope_tags: [code]
- pattern: using # type: ignore to bypass argument type checks instead of using a compatible typed call.
- prevention_rule: prefer call signatures that satisfy typing (for example, typed conversion) and remove ignores in production paths.
- validation_check: in touched files, run rg '# type: ignore' and confirm no new ignores were introduced for the same call path.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:moe_emergence/gpt2_inference.py:164

### MISTAKE-20260226-003
- id: MISTAKE-20260226-003
- status: active
- severity: medium
- scope_tags: [code, typing]
- pattern: assigning plain attributes on nn.Module under strict typing without accounting for custom __setattr__, causing unresolved-attribute diagnostics.
- prevention_rule: for non-parameter/module metadata fields on nn.Module, use object.__setattr__ or another typing-safe pattern and avoid blanket ignores.
- validation_check: run uv run ty check on touched nn.Module files and ensure no unresolved-attribute diagnostics on metadata assignments.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:moe_emergence/gpt2_moe.py:85
  - file:moe_emergence/gpt2_moe.py:86
  - file:moe_emergence/gpt2_moe.py:155

### MISTAKE-20260226-004
- id: MISTAKE-20260226-004
- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: creating model-debate output files in repository root instead of docs/models-debate.
- prevention_rule: when generating model-debate artifacts in this repository, place files under docs/models-debate unless the user explicitly asks for another location.
- validation_check: before writing debate artifacts, verify each output path starts with docs/models-debate/.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 2
- evidence:
  - file:docs/models-debate/011-PHASE5-ANALYSIS-PLAN-CONVERGENCE-2026-02-26.md:1 (moved from repo root after user correction)

### MISTAKE-20260226-005
- id: MISTAKE-20260226-005
- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: documenting dependency changes inconsistently across sections of the same plan (summary table and detailed steps disagree).
- prevention_rule: when plan docs include both a change-summary table and step-by-step instructions, update both sections in the same edit and run a consistency pass before finalizing.
- validation_check: for each listed dependency in detailed steps, verify it appears in the file-modification summary table (or add explicit note that the summary is intentionally partial).
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:docs/project-design/PHASE-5-ANALYSIS-PLAN.md:37 (summary table omits pandas)
  - file:docs/project-design/PHASE-5-ANALYSIS-PLAN.md:48 (step 0 adds pandas)

### MISTAKE-20260226-006
- id: MISTAKE-20260226-006
- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: creating new docs/models-debate files without repository naming convention prefixes (nnn or nnna/nnnb).
- prevention_rule: before creating any new docs/models-debate file, inspect existing filenames and assign the next numeric prefix (for example 011 or 011a/011b) with matching style.
- validation_check: run ls docs/models-debate and confirm each newly created file begins with the intended numeric prefix before finalizing.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:docs/models-debate/011-PHASE5-ANALYSIS-PLAN-CONVERGENCE-2026-02-26.md:1 (renamed from non-prefixed filename after user correction)

### MISTAKE-20260226-007
- id: MISTAKE-20260226-007
- status: active
- severity: medium
- scope_tags: [code]
- pattern: using `Path()` as a sentinel for "no export result" and later treating it as a readable file path.
- prevention_rule: when a function can return "no result", return `None` (not `Path()`), and gate downstream reads with explicit file checks like `is_file()`.
- validation_check: for touched export/io helpers, verify empty-result branches return `None` and downstream consumers guard reads with `path is not None and path.is_file()`.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:scripts/export_wandb.py:54
  - file:scripts/export_wandb.py:88

### MISTAKE-20260226-008
- id: MISTAKE-20260226-008
- status: active
- severity: medium
- scope_tags: [code, planning]
- pattern: implementing a spec-defined manifest without all required contract fields.
- prevention_rule: when implementing plan/spec contracts, build an explicit required-key checklist and verify all required keys are present in serialized outputs.
- validation_check: compare manifest dict keys in touched files against the source-of-truth plan field list before finalizing.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:docs/project-design/PHASE-5-ANALYSIS-PLAN.md:104
  - file:moe_emergence/analysis.py:121

### MISTAKE-20260226-009
- id: MISTAKE-20260226-009
- status: active
- severity: medium
- scope_tags: [code, planning]
- pattern: hard-coding visualization run filters that exclude planned experimental runs.
- prevention_rule: when a plan defines a fixed run set, include that full set in plotting filters unless exclusions are explicitly documented.
- validation_check: for run-comparison plots, verify filter lists match the planned run inventory (`dense`, `moe-main`, `no-lb`, `top-2`).
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:docs/project-design/PHASE-5-ANALYSIS-PLAN.md:173
  - file:moe_emergence/visualize.py:407

### MISTAKE-20260226-010
- id: MISTAKE-20260226-010
- status: active
- severity: medium
- scope_tags: [docs]
- pattern: committing notebook execution artifacts and unrelated metadata churn in a targeted implementation change-set.
- prevention_rule: before finalizing notebook edits, clear outputs/execution counts and restrict metadata changes to task-relevant notebooks only.
- validation_check: for touched notebooks, ensure code-cell outputs are empty, execution counts are null, and metadata diffs are limited to requested scope.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:notebooks/phase5_analysis.ipynb:22
  - file:notebooks/expertweights.ipynb:501

### MISTAKE-20260226-011
- id: MISTAKE-20260226-011
- status: active
- severity: medium
- scope_tags: [code, typing]
- pattern: introducing strict-typing incompatibilities by using runtime-valid but stub-incompatible APIs (for example DataLoader with raw list datasets, mixed-key dict update overloads, or matplotlib attributes/argument container types not reflected in stubs).
- prevention_rule: after touching typed code paths, run `uv run ty check` and resolve stub-incompatible calls using typed wrappers or overload-safe forms before finalizing.
- validation_check: for touched Python files, ensure `uv run ty check <files>` passes and no new `tyinvalid-argument-type`, `tyunresolved-attribute`, or `tyno-matching-overload` diagnostics remain.
- first_seen: 2026-02-26
- last_seen: 2026-02-26
- occurrence_count: 1
- evidence:
  - file:moe_emergence/analysis.py:175
  - file:moe_emergence/analysis.py:213
  - file:moe_emergence/analysis.py:255
  - file:moe_emergence/visualize.py:115
  - file:moe_emergence/visualize.py:200
