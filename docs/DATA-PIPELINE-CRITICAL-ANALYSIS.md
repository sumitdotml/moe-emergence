# Data Pipeline Critical Analysis

**Purpose**: First-principles analysis of the data pipeline before training begins.
**Goal**: Ensure the $80 training budget produces interpretable, defensible results.
**Created**: 2025-01-25

---

## Table of Contents

1. [What Could Invalidate the Experiment?](#what-could-invalidate-the-experiment)
2. [Issue-by-Issue Analysis](#issue-by-issue-analysis)
3. [Revised Severity Assessment](#revised-severity-assessment)
4. [Dataset Verification](#dataset-verification)
5. [Implementation Plan](#implementation-plan)
6. [Decision Summary](#decision-summary)

---

## What Could Invalidate the Experiment?

The V3 spec aims for a "scientifically defensible experimental design" that would satisfy a "Karpathy or Vaswani-level reviewer." Before diving into specific issues, here's what "experiment-invalidating" means for this project:

| Type of Failure | Example | Consequence |
|----------------|---------|-------------|
| **Silent corruption** | Train/eval leakage | Results look good but are meaningless |
| **Confounded signal** | Prefix-based routing | "Specialization" is actually token pattern matching |
| **Biased sample** | Non-representative data | Findings don't generalize even within the domain |
| **Missing validation** | No holdout | Can't distinguish learning from memorization |

---

## Issue-by-Issue Analysis

### 1. CodeParrot Streaming Bias

**GPT-5.2's claim**: Taking the first 10MB of a streaming dataset creates sampling bias. "Code Expert becomes Django Expert."

**First-principles analysis**:

The question isn't whether bias exists—it almost certainly does. The question is: **does it matter for the specific claims being made?**

The claim: "Experts specialize on code vs math vs prose."
NOT the claim: "Experts generalize to all Python code everywhere."

**What would a skeptical reviewer ask?**

> "The results show Expert 3 handles 40% of code tokens. But the code sample is 10MB from an unknown distribution. How do we know Expert 3 learned 'Python patterns' rather than 'Django patterns' or 'unit test patterns' that happen to dominate the sample?"

**The real risk**: If the code sample is homogeneous (e.g., all unit tests, all web frameworks), then:
1. Domain-level heatmaps still work (code ≠ math ≠ prose)
2. Fine-grained Python token analysis still works (keywords vs operators)
3. But claims about "code specialization" are narrower than they appear

**Mitigation options**:

| Option | Effort | Benefit |
|--------|--------|---------|
| Shuffle buffer | 1 line of code | Reduces bias, documented heuristic |
| Document the limitation | 1 paragraph in writeup | Honest framing |
| Manually verify sample diversity | 30 mins of inspection | Know what we actually have |

**Verdict**: Add the shuffle buffer (trivial), and **manually inspect 20-30 samples** from the final dataset to check for variety (web apps, CLI tools, data processing, etc.) vs concerning homogeneity.

**Severity**: MEDIUM. Won't invalidate the experiment, but could narrow the claims.

---

### 2. Math Prefix Pattern

**GPT-5.2's claim**: Router learns to route on `Problem:` token instead of math content. "Cheat emergence."

**Status**: **RESOLVED** — No prefixes for MathQA. See decision 007 (revised).

**Resolution summary**:

After examining 30+ MathQA samples, the decision is to use NO prefixes:
- Format: `{Problem}\n\n{Rationale}`
- MathQA content naturally distinguishes math through numbers, operators (×, /, ⇒), percentages, and step-by-step calculations
- Rationales already vary in their opening phrases ("explanation:", "let x be...", direct calculations)
- The original prefix concern was for invariant "Problem:"/"Solution:" markers — not applicable to MathQA's natural variation

**Post-training test** (still think I should do it):
1. Take a code sample and check routing
2. Verify routing is driven by content, not format artifacts

**Severity**: LOW. Resolved by format choice I believe. Just a FYI.

---

### 3. Train/Eval Data Leakage

**GPT-5.2's claim**: If blocks are split after packing, the same source document spans train and eval.

**First-principles analysis**:

This is the one issue where GPT-5.2 is unambiguously correct, and the severity is accurately rated.

**Why this is a real bug**:

Sequence packing works by concatenating documents:
```
[Doc A tokens] [EOS] [Doc B tokens] [EOS] [Doc C tokens] ...
```

Then chunking into fixed-size blocks:
```
Block 1: [Doc A part 1]
Block 2: [Doc A part 2 | Doc B part 1]
Block 3: [Doc B part 2 | Doc C part 1]
...
```

If blocks are split after packing, Block 2 and Block 3 might end up in different splits, but they share content from Doc B. The model sees Doc B during training and is evaluated on Doc B.

**What this corrupts**:
- Validation loss is artificially low (model has seen the content)
- Any claim about "generalization" is invalid
- Router entropy on validation set is meaningless

**What a reviewer would say**:

> "The validation loss looks good, but there's train/eval leakage. How do we know the model is generalizing rather than memorizing?"

This is a **fundamental experimental flaw**, not a theoretical concern.

**Verdict**: **HIGH severity. Must fix.** Split at the text level before packing. No debate needed.

**Severity**: HIGH. Invalidates validation signal entirely.

---

### 4. Validation Split Design

**GPT-5.2's claim**: Without holdout data, specialization can't be distinguished from memorization.

**First-principles analysis**:

The need for validation is real. The question is: **how much, and how to size it?**

**What validation provides**:

| Metric | Without validation | With validation |
|--------|-------------------|-----------------|
| Training loss | "Model fits training data" | Same |
| Eval loss | N/A | "Model generalizes to unseen data" |
| Train routing patterns | "Router learned something" | Same |
| Eval routing patterns | N/A | "Routing generalizes, not memorized" |

The key claim is: "Experts specialize by domain." Validation enables the stronger statement: "Specialization persists on unseen data" — which is stronger evidence that it's a real learned pattern.

**Holdout sizing options**:

| Approach | Formula | Pros | Cons |
|----------|---------|------|------|
| Fixed count | 50 texts per domain | Simple, predictable | May be too small/large |
| Fixed percentage | 5% of texts | Scales with data size | May yield tiny eval sets |
| Bounded percentage | `min(max(10, n*0.05), n*0.10)` | Handles edge cases | More complex |

**Recommendation**:

Use a simple hybrid approach:
```python
n_eval = max(20, int(len(texts) * 0.05))  # At least 20, target 5%
```

This is simpler than the triple-nested min/max, covers the main edge case (too-small eval), and is easy to explain in the writeup.

**What matters more than the exact formula**:
1. Split at text level before packing (**critical**)
2. Log the actual counts (texts and blocks) (**critical**)
3. Report what was used (**critical**)

**Severity**: MEDIUM. Needed for credibility, but exact sizing formula is less important than doing the split correctly.

---

### 5. Subdomain Provenance

**GPT-5.2's claim**: Can't tell if router splits GSM8K vs MATH (word problems vs LaTeX).

**First-principles analysis**:

This is a pure analysis enhancement, not a correctness issue. Now moot since we're using MathQA only (see Dataset Verification section).

**Severity**: LOW. Nice-to-have for richer analysis, not required for core claims.

---

## Revised Severity Assessment

| Issue | GPT-5.2 | Revised Assessment | Action |
|-------|---------|-------------------|--------|
| Streaming bias | HIGH | MEDIUM | Shuffle buffer proposed — **needs verification** |
| Prefix pattern | HIGH | LOW-MEDIUM | **Needs re-evaluation for MathQA** |
| Train/eval leakage | HIGH | **HIGH** | **Fix before training** |
| Validation split | MEDIUM-HIGH | MEDIUM | Formula proposed — **needs verification** |
| Subdomain labels | MEDIUM | LOW | Not needed with MathQA-only approach |

### Meta-Analysis: Where GPT-5.2 Was Valuable vs Overreached

**Valuable contributions**:
- Caught the train/eval leakage bug (Pass 8) — genuine find
- Pushed for explicit documentation of all heuristics
- Forced justification of recommendations

**Possible overreach**:
- Escalated severities without strong empirical basis
- Assumed worst-case scenarios (streaming order, prefix routing)
- Preferred complex parametric formulas over simple fixed values

**The pattern**: GPT-5.2 optimizes for "defensibility against hypothetical reviewer objections." This is valuable, but can lead to over-engineering when the hypothetical objections are unlikely.

---

## Dataset Verification

### Critical Discovery

**The `hendrycks/competition_math` dataset has a DMCA takedown and is NOT accessible.**

The current `data.py` references this dataset and will fail.

### Dataset Choices Summary

| Domain | Options | Status |
|--------|---------|--------|
| **Code** | CodeParrot-clean OR StarCoderData | **PENDING** — run sample test |
| **Math** | `allenai/math_qa` | **DECIDED** |
| **Prose** | WikiText-103 OR OpenWebText | **PENDING** — run sample test |

---

### Code Dataset Options

#### Option 1: CodeParrot-clean (Current Choice)

| Property | Value |
|----------|-------|
| HF Path | `codeparrot/codeparrot-clean` |
| Size | ~50GB, 5.17M files |
| Language | Python only |
| Access | Open, no TOS |

**Known issues**:
- ~70% of original dataset was duplicated (cleaned version addresses exact duplicates only)
- Near-deduplication not performed
- May contain low-quality or repetitive code

**Verdict**: Convenient but may have quality issues. **Verify samples first.**

#### Option 2: StarCoderData (Alternative)

| Property | Value |
|----------|-------|
| HF Path | `bigcode/starcoderdata` data_dir="python" |
| Size | 783GB total, Python subset TBD |
| Access | Requires TOS acceptance |

**Advantages**: Better cleaning heuristics, more rigorous deduplication, used to train StarCoder.

#### Comparison

| Dataset | Real Code | Quality | Access |
|---------|-----------|---------|--------|
| CodeParrot-clean | Yes | Medium | Open |
| StarCoderData | Yes | Higher | TOS |

---

### Math Dataset — VERIFIED

**Chosen**: MathQA (allenai)

| Property | Value |
|----------|-------|
| Source | https://math-qa.github.io/math-QA/data/MathQA.zip |
| Train Examples | 29,837 |
| Estimated Size | ~11.3MB (exceeds 10MB target) |
| License | Apache 2.0 |
| Fields | `Problem`, `Rationale` |
| Style | Natural language word problems with step-by-step reasoning |

**Note**: The HuggingFace loader (`allenai/math_qa`) seems to have used a deprecated script format and fails. We load directly from the source ZIP file.

**Sample format** (raw from dataset):
```
Problem: A train running at 48 km/hr crosses a pole in 9 seconds. What is the length?
Rationale: Speed = (48 x 5/18) m/sec = (40/3) m/sec. Length = (40/3 x 9) = 120 m.
```

**Training format** (no prefixes — see decision 007 revised):
```
{Problem}

{Rationale}
```

**Why MathQA over alternatives**:
- `hendrycks/competition_math`: DMCA takedown — not accessible
- `nvidia/OpenMathInstruct-1`: ~95% Python code in solutions — blurs code/math distinction
- `openai/gsm8k`: Only ~4MB — below 10MB target
- MathQA: Pure natural language, sufficient volume, legally safe

---

### MathQA Formatting — RESOLVED

**Decision**: No prefixes. Format as `{Problem}\n\n{Rationale}`

**Investigation findings** (2025-01-27):

After examining 30+ MathQA samples:

1. **Content naturally distinguishes math**: Numbers, percentages, operators (×, /, +, -, ⇒), step-by-step calculations
2. **Rationales vary naturally**: Start with "explanation:", "let x be...", direct calculations, etc.
3. **No routing shortcut risk**: Unlike fixed "Problem:"/"Solution:" markers, MathQA's natural variation eliminates this concern

See decision 007 (revised) for full rationale.

---

### Prose Dataset Options

#### Option 1: WikiText-103

| Property | Value |
|----------|-------|
| HF Path | `Salesforce/wikitext` config `wikitext-103-raw-v1` |
| Size | 1.8M rows, ~550MB |
| Content | Wikipedia encyclopedia articles ONLY |

**Concern**: The "prose" domain would actually be "encyclopedia" domain. Wikipedia has a specific style (formal, encyclopedic, structured with headers).

#### Option 2: OpenWebText

| Property | Value |
|----------|-------|
| HF Path | `Skylion007/openwebtext` |
| Size | 41GB (8M documents) |
| Content | Reddit-sourced web content (journalism, blogs, opinions) |
| License | CC0 |

**Advantages**: More diverse writing styles, clearly different from code and math.

#### Comparison

| Property | WikiText-103 | OpenWebText |
|----------|--------------|-------------|
| Diversity | LOW (Wikipedia only) | HIGH (varied sources) |
| Cleanliness | HIGH | MEDIUM |
| Distinctiveness | May overlap with structured text | Clearly different from code/math |

---

### Verified Loading Syntax

```python
from datasets import load_dataset

# Code (Python)
code_ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
# Field: sample["content"]

# Math (MathQA) — loaded from source ZIP, not HuggingFace
# See _load_mathqa_data() in data.py
# URL: https://math-qa.github.io/math-QA/data/MathQA.zip
# Fields: sample["Problem"], sample["Rationale"]

# Prose (WikiText-103)
prose_ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
# Field: sample["text"]

# Prose (OpenWebText) — alternative
prose_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
# Field: sample["text"]
```

---

## Implementation Plan

### Phase 1: Investigation

Before writing any implementation code, these items require investigation and verification.

#### 1.1 MathQA Format Investigation — COMPLETE

**Decision**: No prefixes. Format as `{Problem}\n\n{Rationale}`

See decision 007 (revised) for full investigation results.

#### 1.2 Train/Eval Split Formula Verification

**Goal**: Verify that `max(20, int(n * 0.05))` is appropriate.

**Questions**:
- Where did this formula come from?
- What are the expected text counts per domain at 10MB?
- Does "at least 20" make sense for all domains?
- Should the formula be domain-specific?

#### 1.3 Shuffle Buffer Formula Verification

**Goal**: Verify that `max(1000, size_mb * 200)` is appropriate.

**Questions**:
- Where did this formula come from?
- What buffer size results from 10MB? (10 * 200 = 2000)
- Is this sufficient to reduce streaming bias?
- What does HuggingFace recommend for shuffle buffers?

#### 1.4 Code/Prose Dataset Sample Review

**Goal**: Choose between dataset options based on actual samples.

Run `sample_test.py` and evaluate:
- **Code**: Is CodeParrot diverse? Or dominated by one style?
- **Prose**: WikiText (clean/narrow) vs OpenWebText (diverse/noisier)?

---

### Phase 2: Implementation

After investigations are complete, implement the data pipeline fixes.

**File to modify**: `moe_emergence/data.py`

#### 2.1 Train/Eval Split (REQUIRED — HIGH severity)

Must split at TEXT level before packing. Exact formula TBD after investigation.

#### 2.2 Math Dataset Fix (REQUIRED)

Replace `hendrycks/competition_math` with `allenai/math_qa`. Formatting TBD after investigation.

#### 2.3 Shuffle Buffer (TBD)

Add if investigation confirms it's beneficial. Formula TBD.

---

### Phase 3: Verification

After implementation:

1. **Leakage check**:
   ```python
   assert len(set(train_texts) & set(eval_texts)) == 0
   ```

2. **Balance check**: Log text counts and block counts per domain (train/eval)

3. **Manual inspection**: Look at 20 samples from each domain

---

### Phase 4: Post-Training Verification

**Prefix/format hypothesis test** (adapt based on chosen format):

```python
def test_format_routing_hypothesis(model, moe_modules, tokenizer, code_samples):
    """Test whether math formatting changes routing on non-math content."""
    # Adapt this test based on the chosen math format
    # Goal: verify content drives routing, not format markers
    pass
```

---

## Decision Summary

### Verified

| Decision | Choice | Notes |
|----------|--------|-------|
| Math dataset | `allenai/math_qa` | 29K examples, ~12MB, Apache 2.0, legally safe |
| Train/eval leakage fix | Split at TEXT level before packing | HIGH severity, must implement |

### Pending Investigation

These proposals came from the multi-model debate. They need independent verification before implementation.

| Item | Proposed Approach | Status |
|------|-------------------|--------|
| MathQA formatting | `{Problem}\n\n{Rationale}` (no prefixes) | **RESOLVED** — see decision 007 |
| Train/eval split formula | `max(20, int(n * 0.05))` | Pending verification |
| Shuffle buffer | `max(1000, size_mb*200)` | Pending verification |
| Code dataset | CodeParrot-clean | Pending sample review |
| Prose dataset | WikiText-103 or OpenWebText | Pending sample review |

### Post-Training

| Item | Purpose |
|------|---------|
| Prefix hypothesis test | Verify content vs format drives routing |

---

## Current data.py Issues

| Line | Issue | Fix Required | Status |
|------|-------|--------------|--------|
| 140-205 | Uses GSM8K + `hendrycks/competition_math` | Replace with MathQA from ZIP | **BLOCKING** — dataset is DMCA'd |
| N/A | No train/eval split | Add text-level split before packing | **BLOCKING** — HIGH severity |
| 113 | No shuffle buffer for CodeParrot | TBD after investigation | Pending |
| — | Fixed prefixes | No prefixes for MathQA | **RESOLVED** — see decision 007 |

---

## Verification Commands

After implementing fixes:

```bash
uv run python moe_emergence/data.py --size-mb 10 --block-size 512 --seed 42
```

Expected output:
- Separate train/eval counts per domain
- No leakage assertion errors
- Shuffle buffer size logged
