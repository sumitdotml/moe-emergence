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
