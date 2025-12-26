# GPT-5.2 PASS 1

• Opus Review Findings

- MEDIUM: The claim that specialization is "unfalsifiable" without a held-out set overstates the impact; the pipeline
  explicitly says evaluation is separate, so this should be framed as a generalization limitation rather than
  invalidating results. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:9 docs/DATA-PIPELINE.md:13
- MEDIUM: The "wrong direction" truncation critique ignores the explicit decision to avoid oversampling to prevent
  repetition-driven specialization; the review should acknowledge that tradeoff before recommending oversampling.
  docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:65 docs/decisions/005-phase3-data-sizing.md:52 docs/DATA-PIPELINE.md:120
- LOW-MEDIUM: "Filter to Python-only" is not actionable as written because CodeParrot streaming has no language label,
  so this implies extra tooling or heuristics and added scope that the review does not mention. docs/DATA-CRITICAL-
  REVIEW-OPUS.md:35 docs/DATA-PIPELINE.md:21
- LOW-MEDIUM: The MATH-only suggestion does not check feasibility against the documented 10MB target and the stated
  reason for combining GSM8K + MATH. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:49 docs/DATA-PIPELINE.md:56
- LOW: The tail-drop nondeterminism point should be scoped to the streaming code dataset; math and prose are non-
  streaming. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:78 docs/DATA-PIPELINE.md:21

Data Pipeline Findings (My Review)

- HIGH: CodeParrot is streamed and never shuffled, so you take the first N MB, which can bias language/topic mix and
  distort domain balance; for small targets this is a real representativeness risk. moe-emergence/data.py:113 moe-
  emergence/data.py:119
- MEDIUM: The fixed "Problem:" and "Solution:" prefixes create a strong lexical shortcut for math; routing may
  specialize on the prefix rather than math content, so consider ablation or prefix removal. moe-emergence/data.py:147
  docs/DATA-PIPELINE.md:44
- MEDIUM: Subdomain provenance is discarded at block level (GSM8K vs MATH, code language), so you cannot tell whether
  experts split by subsource rather than by domain; add a subdomain tag or per-block origin logging. moe-emergence/
  data.py:85 moe-emergence/data.py:196
- MEDIUM: No validation slice is defined anywhere beyond "evaluation is separate"; without even a small holdout, any
  specialization narrative is harder to defend. docs/DATA-PIPELINE.md:13
- LOW-MEDIUM: Token balancing is optional and off by default; given known tokenization skew, this should be a
  conscious, logged choice for each run. docs/DATA-PIPELINE.md:116 moe-emergence/data.py:322
- LOW: WikiText-103 bias is already documented; if unchanged, the report should explicitly frame results as
  "encyclopedic prose" rather than general prose. docs/DATA-PIPELINE.md:61 docs/DATA-PIPELINE.md:197

Cross-check

- Overlap: both reviews flag validation split risk, prose source bias, code language mix, and GSM8K/MATH
  heterogeneity. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:9 docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:21 docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:35 docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:49
- Opus-only: per-domain loss tracking and tail-drop nondeterminism emphasis, plus the oversampling recommendation.
  docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:65 docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:86
- Mine-only: streaming sampling bias, math prefix leakage, subdomain label loss, and the default-off balancing risk.
  moe-emergence/data.py:113 moe-emergence/data.py:147 moe-emergence/data.py:85 docs/DATA-PIPELINE.md:116

Open Questions

- Do you want to keep the "Problem/Solution" prefix or run a small ablation to test routing without it?
- Do you want a tiny validation split (2-5%) per domain to defend against memorization claims?
- Is preserving GSM8K vs MATH and code language labels worth the extra bookkeeping for analysis?

Change Summary

- No files changed; no tests run.

If you want me to act on this, here are the clearest next steps:

1. Decide whether to add a small validation split and where to document it.
2. Decide whether to shuffle the streamed CodeParrot data (and how) to reduce sampling bias.
3. Decide whether to preserve subdomain labels and/or drop the math prefixes for analysis.

---

# GPT-5.2 PASS 2

Reviewed the updated CLAUDE OPUS 4.5 PASS 2; no web search used, so external citations in the doc are not verified by me.

CLAUDE OPUS 4.5 PASS 2 Quality Issues

- The new citations are unverified and their relevance to top‑1 routing is assumed; treat the quote as a hypothesis or
  remove it until verified. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:167 docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:267
- The claim about a non‑streaming “codeparrot-clean-train” split may be inaccurate; this needs confirmation before it
  becomes guidance. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:159
- “Make --balance-tokens default” conflicts with the accepted decision that balancing is optional to avoid truncation;
  if this changes, it must be a new decision and reflected in docs. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:203 docs/
  decisions/005-phase3-data-sizing.md:47
- “Math prefix leakage” is plausible but framed as a strong conclusion; it should be explicitly labeled a tradeoff or
  supported with a small ablation, since prefixes are also a deliberate domain signal. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:165 moe-emergence/data.py:170

Gaps / Underweighted

- The proposed “load 2x then shuffle” still samples from the start of the stream; you’ll want a streaming shuffle
  buffer or reservoir sampling call‑out to reduce ordering bias. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:157
- The 5% validation split recommendation is reasonable but may be heavy for 10MB; consider 2–5% or fixed eval prompts
  and document how “separate evaluation” is actually done. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:246 docs/DATA-
  PIPELINE.md:13
- Subdomain labeling for code is proposed but no plan is given; either scope it to math only or define a concrete
  language ID approach. docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md:244

Decisions To Lock Before Training

- Streaming bias mitigation: pick one path: 1) streaming shuffle buffer/reservoir sampling, 2) non‑streaming load +
  shuffle + truncate, 3) accept bias and explicitly document it.
- Math prefixes: pick one: 1) remove prefixes, 2) randomize prefix strings, 3) keep prefixes but exclude first N
  tokens in analysis.
- Validation: pick one: 1) 2–5% per domain holdout, 2) fixed prompt eval set only, 3) no holdout but explicit “no
  generalization claims.”
- Subdomain labeling: pick one: 1) add subdomain for GSM8K/MATH only, 2) add subdomain + code language ID, 3) keep
  single domain labels and document the limitation.

If you decide on any of these, I recommend logging it via /doc-decision so the audit trail stays intact.

---

# GPT-5.2 PASS 3

Scope

- Reviewed: docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md (PASS 3), docs/models-debate/005c-DATA-PIPELINE-CRITICAL-REVIEW-GEMINI-3.md, docs/DATA-PIPELINE.md,
  docs/decisions/005-phase3-data-sizing.md, moe-emergence/data.py.
- No web search used; external citations in other reviews remain unverified.

Critical Review of Gemini 3 Pass 1

Strengths

- Correctly elevates streaming bias and math prefix leakage as experiment-invalidating risks.
- Flags validation split and subdomain provenance loss as credibility risks.

Issues / Overreach

- Uses unsourced numbers (e.g., "180GB", "first 50 repos") and "alphabetical trap" framing without evidence.
- Recommends "buffer_size=100\*target_size" without memory/latency analysis or a concrete streaming shuffle method.
- Recommends default token balancing, which conflicts with the accepted decision doc; changing this needs a new decision.
- Ends mid-sentence and reads like a draft; treat as advisory notes, not a finalized spec.

Critical Review of Opus Pass 3

Improvements

- Correctly retracts earlier flawed fixes and centers the right blockers (streaming bias, prefix leakage).
- Provides concrete mitigation options and decision framing.

Remaining Issues

- Still proposes "add 5% validation split" as default; earlier analysis suggests 2-5% is safer at 10MB scale.
- External citations are still unverified and should be labeled as hypotheses until checked.
- The suggested shuffle buffer size is unspecified; an undersized buffer can still be biased.

Where I Land (Final Stance)

Blockers Before Any Training Run

1. CodeParrot sampling bias must be addressed. The current "first N MB" stream is not acceptable.
   - Preferred: streaming shuffle buffer (datasets .shuffle(buffer_size=...)) or reservoir sampling.
   - Minimum: explicitly document "first N MB" bias if you choose to accept it.
2. Math prefix leakage must be mitigated.
   - Preferred: remove fixed "Problem:"/"Solution:" prefixes or randomize them; do not keep a single invariant token.
   - If kept, analysis must exclude the prefix tokens.
3. Add a validation signal.
   - Preferred: 2-5% per-domain holdout or a fixed eval prompt set; document which and why.

Strongly Recommended (Non-Blocking but High Value) 4) Preserve math subdomain labels (gsm8k vs competition_math) in dataset outputs. 5) Keep token balancing opt-in per decision doc, but require explicit run logging ("balance_tokens: true/false").

Low Priority / Document-Only 6) WikiText-103 prose bias should be acknowledged in results as "encyclopedic prose."

Decisions That Must Be Locked (No Training Until Decided)

- Streaming mitigation: shuffle buffer vs reservoir sampling vs accept bias and document.
- Math prefix handling: remove vs randomize vs keep and exclude in analysis.
- Validation strategy: small holdout vs fixed eval prompts vs no holdout + explicit disclaimer.
- Subdomain tracking: math only vs math + code language ID vs none.

If you want, I can implement the chosen options and add a /doc-decision entry for each.

---

# GPT-5.2 PASS 4

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 4 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; any external defaults or citations are treated as unverified.

Critical Review of Opus Pass 4

Strengths

- Cleanly consolidates blockers and adopts the "decisions must be locked" framing.
- Explicitly corrects earlier flawed mitigation suggestions (load 2x then shuffle).
- Keeps balance_tokens opt-in and ties any change to a formal decision.

Issues / Assumptions to Tighten

- Assumes a default datasets shuffle buffer size (1000) without verification; treat as a guess or confirm.
- Uses a back-of-envelope "10MB -> ~600 blocks" estimate that ignores tokenization variance and per-domain skew.
- Recommends buffer_size=10000 without linking to target sample size or memory budget; this should be justified or
  parameterized.
- Suggests "randomize prefixes" but keeps a fixed "Solution:" token, which can still become a routing shortcut.
- Treats the final recommendations as effectively decided; they are still proposals until logged via /doc-decision.

Where I Land Now

Blockers remain unchanged:

1. Fix CodeParrot sampling bias (streaming shuffle buffer or reservoir sampling, not "first N MB").
2. Mitigate math prefix leakage (remove or randomize prefixes; if kept, exclude prefix tokens in analysis).
3. Add a validation signal (2-5% holdout or fixed eval prompts; document which).

Refinements I would make to Pass 4 recommendations:

- Buffer sizing should be tied to expected example count or token budget, not a fixed constant.
- If randomizing math prefixes, randomize both problem and solution markers or remove all fixed markers.
- Subdomain labels for math are low effort and high value; code language ID should be optional, not required.

Decision Reminder

No training until these are explicitly decided and documented:

- Streaming mitigation method and buffer strategy.
- Prefix handling strategy.
- Validation strategy.
- Subdomain tracking scope.

---

# GPT-5.2 PASS 5

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 5 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; all external defaults remain unverified.

Critical Review of Opus Pass 5

Strengths

- Correctly fixes the "Solution:" marker oversight and tightens prefix handling options.
- Moves buffer sizing from a constant to a size‑dependent rule of thumb.
- Keeps code language ID optional and math subdomains prioritized.

Issues / Assumptions to Tighten

- The "expected_examples ~ size_mb \* 1000" heuristic is unstated but implied; should be explicit or avoided.
- Buffer rule ">=10% of expected samples" is plausible but unvalidated; a smaller or larger buffer may still be biased.
- Recommends "remove all markers" as default without noting it changes data format and may affect downstream analysis
  (e.g., inability to locate solution spans).
- Validation recommendation still assumes a uniform %; for small datasets, a fixed‑count holdout may be more stable.

Where I Land Now

Blockers remain:

1. Streaming bias mitigation (shuffle buffer or reservoir sampling) must be implemented or explicitly documented.
2. Math prefix leakage must be mitigated (remove markers, randomize both markers, or exclude marker tokens in analysis).
3. Validation signal must exist (2‑5% holdout or fixed eval set with clear documentation).

Refinements to Opus Pass 5 Recommendations

- Buffer size should be parameterized and logged, not hardcoded; consider a CLI flag like --shuffle-buffer.
- If removing markers, add a minimal separator token (e.g., newline) and document that solution span is implicit.
- Prefer a fixed minimum holdout size (e.g., 50 blocks) plus a cap (e.g., 2‑5%) to avoid tiny eval sets.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:

- Streaming mitigation method and buffer sizing strategy.
- Marker handling (remove vs randomize vs exclude).
- Validation strategy (percent vs fixed count vs prompts).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 6

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 6 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; defaults and numeric estimates remain unverified unless documented in-code.

Critical Review of Opus Pass 6

Strengths

- Correctly accepts the marker-boundary tradeoff and moves to randomizing both markers.
- Recognizes buffer sizing uncertainty and proposes making it explicit/logged.
- Keeps the focus on the same three blockers and avoids scope creep.

Issues / Assumptions to Tighten

- Defaulting to `--shuffle-buffer 5000` is arbitrary unless tied to data size; should be parameterized or justified.
- The "always 5% holdout is fine at 10MB" assumes stable block counts; tokenization skew can make 5% too small for
  one domain and too large for another.
- "Log prefix pairs per example" adds bookkeeping; ensure it is actually used in analysis or skip it to keep scope tight.
- Suggests a CLI flag without confirming whether streaming shuffle supports it for the dataset version in use.

Where I Land Now

I agree with the convergence on the blockers, but I would keep defaults conservative and parameterized:

1. Streaming bias: add a shuffle buffer with explicit size and log it; choose a default that scales with size_mb.
2. Prefix leakage: randomize both markers or remove them; if randomizing, keep a small controlled set and document it.
3. Validation: prefer a hybrid rule (min blocks + percent cap) to avoid tiny eval sets in any domain.
4. Subdomain: math-only sublabels are worthwhile; defer code language ID unless analysis needs it.

Refined Recommendations

- Buffer sizing: make `--shuffle-buffer` optional but default to `max(1000, size_mb * 200)` or similar simple scaling,
  and log the final value.
- Holdout sizing: use `max(20, int(blocks * 0.05))` per domain with a cap (e.g., 10%) to prevent tiny or huge eval sets.
- Prefix randomization: randomize both problem and solution markers, and optionally store the chosen pair in metadata
  only if you will use it in analysis.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:

- Streaming mitigation method and buffer sizing rule.
- Marker handling strategy (remove vs randomize vs exclude).
- Validation strategy (percent vs fixed vs hybrid).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 7

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 7 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; defaults and dataset API support remain unverified unless checked in-code.

Critical Review of Opus Pass 7

Strengths

- Calls out analysis paralysis and pushes toward decision lock‑in.
- Keeps the blockers stable and avoids new scope creep.
- Simplifies implementation choices to reduce engineering drag.

Issues / Assumptions to Tighten

- Fixed buffer size (5000) is arbitrary and not tied to data size or memory budget; could be too small or too large.
- "5% or 25 blocks minimum" can overshoot small domains (e.g., 300 blocks -> 25 = 8.3%) and bias evaluation.
- Skipping prefix logging is fine only if you truly will not analyze prefix effects; otherwise it becomes irreversible.
- Assumes streaming shuffle works as proposed; still needs verification against the actual datasets version.

Where I Land Now

I agree we should lock decisions soon, but prefer a light parameterization over fixed constants:

1. Streaming bias: use a shuffle buffer with a size that scales with `size_mb` and is logged.
2. Prefix leakage: randomize both markers from a small controlled set; logging is optional but cheap if you want
   later analysis.
3. Validation: use a hybrid rule to avoid tiny eval sets but cap the fraction to avoid large holdouts in small domains.
4. Subdomain: math-only sublabels are still the best value/effort tradeoff.

Refinements I Would Keep

- Buffer sizing: `max(1000, size_mb * 200)` is simple, scales with size, and avoids a hardcoded constant.
- Holdout sizing: `min(max(20, int(blocks * 0.05)), int(blocks * 0.10))` is a reasonable compromise.
- Prefix logging: optional; include only if you intend to analyze prefix effects.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:

- Streaming mitigation method and buffer sizing rule.
- Marker handling strategy (remove vs randomize vs exclude).
- Validation strategy (percent vs fixed vs hybrid).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 8

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 8 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; any "verified" API claims remain unconfirmed unless tested locally.

Critical Review of Opus Pass 8

Strengths

- Converges on the same blockers and accepts parameterized formulas.
- Tightens prefix handling and removes logging to keep scope lean.
- Adds an explicit pre-implementation checklist.

Issues / Assumptions to Tighten

- The "Verified" claim for streaming shuffle is still not actually verified here; it is a doc-level assertion.
- The buffer formula assumes a rough examples-per-MB ratio; it is a heuristic, not a guarantee of unbiased sampling.
- Randomizing with empty prefixes removes boundary cues; if boundary analysis is desired later, this becomes irreversible.
- Holdout sizing uses block counts, but the split strategy is undefined (before or after packing), which affects leakage.

New Gap to Consider

- Holdout should be created at the text/document level before packing to avoid train/eval leakage from the same source
  sequence being split across blocks. If holdout is done after packing, leakage is likely.

Where I Land Now

We are converged on the core decisions, but I would add two implementation constraints:

1. Shuffle buffer should be seeded and logged for reproducibility.
2. Holdout split should happen before packing (text-level), then pack train and eval separately.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:

- Streaming mitigation method and buffer sizing rule (including seed).
- Marker handling strategy (remove vs randomize vs exclude, and whether empty markers are allowed).
- Validation strategy (percent vs fixed vs hybrid, and split level: text vs block).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 9

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 9 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; any "verified" API claims are not confirmed unless executed locally.

Critical Review of Opus Pass 9

Strengths
- Correctly identifies text-level split before packing as essential to avoid leakage.
- Adds shuffle seeding for reproducibility.
- Keeps the core decision set stable and implementation‑focused.

Issues / Assumptions to Tighten
- Holdout sizing is still expressed in blocks while splitting happens in text space; decide whether holdout targets
  text counts or block counts and document the tradeoff.
- The "Verified" shuffle API claim remains doc‑level until run; keep it as a checklist item, not a conclusion.
- Randomizing prefixes with empty strings weakens boundary cues; if later analysis needs boundaries, avoid empty markers
  or add a minimal delimiter token.
- For streaming data, text-level split plus size-based truncation can skew eval size; clarify whether size_mb applies
  to train only or to train+eval combined.

New Constraint to Preserve Correctness

- Train/eval must be packed separately per domain after a text-level split to avoid cross‑split mixing in packed blocks.

Where I Land Now

We are converged on the blockers, with one clarity add:
1) Define holdout in text space first, accept resulting block counts, and log them explicitly.
2) Treat shuffle support and determinism as verified only after a local check.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:
- Streaming mitigation method and buffer sizing rule (including seed).
- Marker handling strategy (remove vs randomize vs exclude; whether empty markers are allowed).
- Validation strategy (text‑level split definition and sizing target).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 10

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 10 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; "verified" API claims remain checklist items unless executed.

Critical Review of Opus Pass 10

Strengths
- Resolves the holdout sizing ambiguity by defining it in text space.
- Removes empty prefixes to preserve boundary cues.
- Clarifies `size_mb` as pre‑split total and logs both text and block counts.
- Adds explicit guidance for streaming + shuffle + text‑level split workflow.

Issues / Assumptions to Tighten
- The holdout formula uses text counts but still assumes a minimum of 20 texts; for very small domains this may be too
  aggressive. Consider a minimum of 10 texts or a min percentage when data is tiny.
- The buffer size formula remains heuristic; document that it is a bias‑reduction knob, not a guarantee.
- The "take from end after shuffle" split should be explicit (randomized ordering), otherwise split order matters.
- For streaming datasets, size_mb as total pre‑split implies eval size shrinks proportionally; document the expected
  eval size in tokens/blocks to avoid surprise.

Where I Land Now

We are converged enough to implement, with two guardrails:
1) Ensure the text list is shuffled deterministically before train/eval split.
2) Treat buffer size and holdout size as logged parameters for every run.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:
- Streaming mitigation method and buffer sizing rule (including seed).
- Marker handling strategy (remove vs randomize vs exclude; empty markers disallowed).
- Validation strategy (text‑level split definition and sizing target).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 PASS 11

Scope

- Reviewed: CLAUDE OPUS 4.5 PASS 11 in docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md.
- No web search used; any "verified" claims remain checklist items unless executed.

Critical Review of Opus Pass 11

Strengths
- Accepts deterministic text shuffle before split and explicit parameter logging.
- Tightens holdout minimum to 10 texts for tiny domains.
- Clearly enumerates the critical issues resolved and proposes concrete next steps.

Issues / Assumptions to Tighten
- Holdout minimum of 10 texts is still arbitrary; document as a heuristic, not a guarantee.
- "Log prefix choices in every run" conflicts with the earlier decision to avoid per‑example prefix logging; clarify
  whether this means logging the prefix sets only (recommended) or logging per‑example choices (scope creep).
- Declares formal convergence before decisions are logged via /doc-decision; these are still proposals until recorded.
- Size_mb semantics are set to pre‑split total; confirm this is acceptable for downstream comparability with earlier
  logs or update docs accordingly.

Where I Land Now

We are converged enough to implement, with two clarifications:
1) Keep prefix logging to the configured prefix sets, not per‑example choices, unless you plan prefix‑specific analysis.
2) Treat min holdout and buffer size rules as documented heuristics and log their computed values for each run.

Decision Reminder

No training until these are explicitly decided and logged via /doc-decision:
- Streaming mitigation method and buffer sizing rule (including seed).
- Marker handling strategy (randomize both; no empty markers; logging scope).
- Validation strategy (text‑level split definition, min holdout rule, and sizing target).
- Subdomain tracking scope (math only vs expanded).

---

# GPT-5.2 FINAL PASS

Final Recommendations (Locked)

1) Streaming mitigation
   - Use streaming shuffle with seed: `ds.shuffle(buffer_size=max(1000, size_mb * 200), seed=args.seed)`.
   - Log buffer_size and seed for every run.
   - Treat buffer_size as a heuristic bias-reduction knob, not a guarantee.

2) Math prefix handling
   - Randomize both markers from fixed non-empty sets:
     - Problem: ["Problem:", "Question:", "Given:"]
     - Solution: ["Solution:", "Answer:", "Therefore:"]
   - Do NOT log per-example prefix choices; only log the prefix sets and seed.

3) Validation strategy
   - Split at text/document level before packing.
   - Deterministically shuffle texts with seed before split.
   - Holdout sizing in text space:
     `n_eval_texts = min(max(10, int(n_texts * 0.05)), int(n_texts * 0.10))`
   - Pack train and eval separately per domain; log text counts and resulting block counts.
   - `--size-mb` applies to total (train+eval) pre-split; document that eval is ~5% of total.

4) Subdomain tracking
   - Add math-only subdomain labels: "gsm8k" vs "competition_math".
   - Defer code language ID until analysis demands it.

5) Documentation/Logging
   - Keep `--balance-tokens` opt-in per decision doc; always log the choice.
   - Document WikiText-103 as "encyclopedic prose" in results.

Implementation Guardrails

- Verify streaming shuffle API locally before claiming it is "verified".
- Ensure no cross-split leakage by packing train/eval separately.
- Log all computed parameters (buffer_size, holdout sizes, seed, prefix sets).

Ready to proceed to /doc-decision entries and implementation.
