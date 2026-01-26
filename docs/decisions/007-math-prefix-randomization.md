# Decision: Math Data Formatting (No Prefixes)

**Date:** 2025-01-21 (original), 2025-01-27 (revised)
**Status:** Revised
**Context Commit:** `4b7b45e` (original), `e18bea0` (revision)

---

## Revision Note (2025-01-27)

This decision was originally written for GSM8K + hendrycks/competition_math datasets. Due to the DMCA takedown of hendrycks/competition_math and the deprecation of the allenai/math_qa HuggingFace loader, we now load MathQA directly from the source ZIP file.

After examining 30+ MathQA samples, the original prefix randomization approach is **no longer needed**. See "Revised Decision" below.

---

## Original Context

The original math data loader formatted every example as:

```
Problem: {question}

Solution: {answer}
```

This created a routing shortcut concern: MoE routers look for the strongest signal. The token `Problem:` at the start of every math sequence could be a stronger signal than the mathematical content itself.

---

## Revised Decision: No Prefixes for MathQA

**Format:** `{Problem}\n\n{Rationale}` (no prefixes)

**Rationale for removing prefixes:**

1. **Natural content distinguishes math.** MathQA samples contain abundant mathematical signals:
   - Numbers and percentages throughout ("120 is what percent of 50?")
   - Arithmetic operators (×, /, +, -, ⇒)
   - Step-by-step calculations ("40 \* x = 120 --> x = 2.4")
   - Mathematical vocabulary ("ratio", "average", "percent", "find")

2. **Rationales already vary naturally.** Unlike fixed "Solution:" prefixes, MathQA rationales begin with varied phrases:
   - "explanation: ..."
   - "let x be the total..."
   - "50 \* x = 120 --> ..."
   - Direct calculations without preamble

3. **Simpler implementation.** For an $80 demo project, avoiding unnecessary complexity is valuable.

4. **The original concern was for fixed markers.** GSM8K/MATH had invariant "Problem:"/"Solution:" at every example start. MathQA's natural variation eliminates this specific risk.

**Sample format:**

```
the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?

explanation : t = 3 years r = 10 % td = ( bg × 100 ) / tr = ( 36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs . 120 td = ( pw × tr ) / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = 1200 / 3 = rs . 400 answer : option a
```

---

## Original Options Considered (Historical)

### Option A: Keep Fixed Prefixes

- **Rejected:** Router learns prefix tokens, not math content

### Option B: Remove All Prefixes

- **Now Adopted** (with MathQA's natural content variation)

### Option C: Randomize Both Prefixes

- **Originally Accepted** for GSM8K/MATH
- **No longer needed** for MathQA

### Option D: Keep Prefixes, Exclude in Analysis

- **Rejected:** Post-hoc workaround

---

## Consequences

**Positive:**

- Simpler data pipeline
- Router must learn from mathematical content, not artificial markers
- Natural variation in rationale openings provides implicit "randomization"

**Negative:**

- No explicit problem/solution boundary marker (minor — the double newline serves as separator)

**Risks:**

- If router still finds shortcuts, we can revisit. But MathQA's content density makes this unlikely.

---

## Unicode Mathematical Symbols in MathQA

MathQA contains **Unicode mathematical symbols** (not LaTeX). These are preserved as-is, so no conversion needed.

**Common symbols in the dataset:**

| Symbol | Unicode | Name                    | ASCII equivalent |
| ------ | ------- | ----------------------- | ---------------- |
| ×      | U+00D7  | Multiplication sign     | \*               |
| ⇒      | U+21D2  | Rightwards double arrow | =>               |
| ⋅      | U+22C5  | Dot operator            | \* or .          |
| −      | U+2212  | Minus sign              | -                |

**Why we don't convert them:**

1. **Tokenizer handles them natively.** GPT-2's tokenizer encodes and decodes Unicode symbols correctly; round-trip works.

2. **Different tokens from ASCII.** `×` (token 13958) and `*` (token 1635) are distinct tokens. The model sees them as different.

3. **No impact on domain separation.** MathQA uses both ASCII `*` (30,174 occurrences) and Unicode `×` (3,777 occurrences). There's no clean symbol-based separation from code. Domains are distinguished by content patterns (word problems, mathematical vocabulary, reasoning chains for math; syntax, imports, function definitions for code).

**Verification:**

```python
tokenizer.encode('3 × 10')  # [18, 13958, 838] — Unicode ×
tokenizer.encode('3 * 10')  # [18, 1635, 838]  — ASCII *
```

**Why this probably doesn't matter for our goals:**

The mixed `*`/`×` usage means this symbol appears in both math and code domains. Could this confuse the router maybe? For my specific experiment, probably no (obviously does affect, but is a bit of a marginal gain that doesn't quite matter at this scale):

1. **Router sees context, not isolated tokens.** At layers 8-11, a `*` in `def f(a,b): return a * b` has a completely different hidden representation than `*` in `"40 * x = 120 so x = 3"`. Surrounding tokens contextualize it.

2. **Domain signals are much stronger.** Math has word problems, "find the value", "answer: a", reasoning chains. Code has `def`, `import`, indentation, syntax. These structural differences dwarf symbol choice.

3. **I'm testing coarse-grained specialization.** The experiment asks "do experts route differently for math vs code vs prose?" and not "can the model distinguish multiplication notations?" :)

**When would this matter?**

- Fine-grained analysis of operator handling
- Claims about the model "understanding" mathematical notation
- If symbol usage correlated with a confounding variable
- Maybe for a much larger scale where marginal gains matter the most

**Conclusion:** Not a meaningful caveat for this $80 demo project. The inconsistency exists but won't affect my coarse-grained domain specialization conclusions.

**Data quality notes:**

- Some rationales have quotes around them (`"explanation : ..."`), others don't
- Some samples have garbled text (e.g., `23 ⋅ x 23 ⋅ x` appears to be a copy-paste error for fractions)
- I'm accepting these as-is. Cleaning would add complexity for marginal benefit

---

## References

- Original debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md`
- MathQA source: https://math-qa.github.io/math-QA/data/MathQA.zip
- Dataset paper: Amini et al., "MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms"
