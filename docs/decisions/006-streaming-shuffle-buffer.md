# Decision: Streaming Shuffle Buffer for CodeParrot

**Date:** 2025-01-21
**Status:** Investigated — Not Implemented
**Context Commit:** `4b7b45e`
**Investigation Date:** 2026-02-05

---

## Context

CodeParrot is loaded via streaming (`streaming=True`) and the current implementation takes the first N MB of data by breaking after reaching the target size. This creates a sampling bias: streaming datasets iterate in a fixed order (by repo ID, file size, or index position), so the first 10MB is not representative of the full dataset.

This was identified as a HIGH severity issue in the multi-model debate (Opus 4.5 + GPT-5.2 + Gemini 3, 11 passes). Gemini 3 called it the "Alphabetical Trap" — the code domain might actually be "Django Expert" or "Unit Test Expert" depending on what repos appear first.

---

## Options Considered

### Option A: Accept Bias and Document

**Description:** Keep current behavior, document the limitation.

**Pros:**

- No code changes needed
- Simplest approach

**Cons:**

- Invalidates "code specialization" claims
- Results are artifacts of dataset ordering, not true domain patterns

### Option B: Load 2x, Shuffle in Memory, Truncate

**Description:** Load more data than needed, shuffle in memory, then truncate.

**Pros:**

- Simple to implement

**Cons:**

- Still samples from the START of the stream — doesn't fix the bias
- Memory overhead

### Option C: Streaming Shuffle Buffer

**Description:** Use `ds.shuffle(buffer_size=N, seed=S)` to shuffle while streaming.

**Pros:**

- Proper randomization without loading entire dataset
- Memory-efficient (only buffer_size examples in memory)
- Reproducible with seed
- Supported by HuggingFace `datasets` library

**Cons:**

- Buffer size is a heuristic, not a guarantee of unbiased sampling
- Larger buffers use more memory

### Option D: Reservoir Sampling

**Description:** Implement true reservoir sampling for uniform random selection.

**Pros:**

- Theoretically optimal for random sampling from stream

**Cons:**

- More complex to implement
- Overkill for this project's scale

---

## Original Decision (2025-01-21)

**Option C: Streaming Shuffle Buffer** with the following parameters:

```python
buffer_size = max(1000, int(size_mb * 200))
ds = ds.shuffle(buffer_size=buffer_size, seed=args.seed)
```

**Rationale:**

- Balances randomization quality with simplicity
- Buffer scales with target size (at 10MB: buffer=2000)
- Seed ensures reproducibility
- Formula assumes ~200 examples per MB (rough heuristic for filtered CodeParrot)

---

## Empirical Investigation (2026-02-05)

Before implementing the shuffle buffer, we investigated whether the "alphabetical trap" concern was valid by analyzing actual samples from both streaming datasets (CodeParrot and C4).

### CodeParrot Analysis (500 samples)

**Question:** Is the data heavily clustered by project type, or already diverse?

**Method:** Analyzed first 500 samples for framework/library usage patterns.

**Sample examples from sequential stream:**

| Position | Content Preview                                                     | Type       |
| -------- | ------------------------------------------------------------------- | ---------- |
| 0        | `Copyright (C) 2013-2014 Tavendo GmbH... Apache License`            | asyncio    |
| 1        | `from django.utils.itercompat import is_iterable`                   | Django     |
| 2        | `"""The :mod:`sklearn.utils` module includes various utilities."""` | sklearn    |
| 3        | `""" Python Character Mapping Codec cp1250 generated from...`       | stdlib     |
| 4        | `#!/usr/bin/python... Copyright: (c) 2013, Matthias Vogelgesang`    | Ansible    |
| 20       | TensorFlow/argparse code                                            | tensorflow |
| 40       | `from javascript import JSObject`                                   | other      |
| 100      | Django code                                                         | Django     |

**Framework distribution (first 200 samples):**

```
other           114 (57%)
django           31 (15.5%)
numpy            22 (11%)
unittest         17 (8.5%)
requests          7 (3.5%)
sklearn           5 (2.5%)
tensorflow        5 (2.5%)
flask             2 (1%)
pandas            1 (0.5%)
```

**Django position analysis (500 samples):**

- Django files found: 81 out of 500 (16.2%)
- Positions: `[1, 8, 11, 13, 22, 28, 36, 44, 46, 57, 62, 73, 74, 85, ...]`
- Distribution is scattered throughout, NOT concentrated at start

Visual distribution (D=Django, .=other):

```
  0- 49: .D......D..D.D........D.....D.......D.......D.D...
 50- 99: .......D....D..........DD..........D.....D....D...
100-149: DD.......D.....D...D.D..........D.D...D...........
150-199: ................................DDDD....D........D
200-249: ......D.D..D.D.......D............D..D.D....DDD...
```

**Clustering analysis:**

- Maximum consecutive Django files: 4 (positions 182-185)
- Average gap between Django files: 6.2 samples
- No severe clustering detected

### C4 (Prose) Analysis (200 samples)

**Question:** Does C4 show topic clustering that could bias the prose domain?

**Sample examples from sequential stream:**

| Position | Content Preview                                                                             |
| -------- | ------------------------------------------------------------------------------------------- |
| 0        | "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making..."      |
| 20       | "A full time vegan, poet, artist and woman. Frontal and thorough when inspired..."          |
| 40       | "Volunteers are needed to help with the Assembly for Children at the 99th International..." |
| 60       | "Everyone seems to want a brighter smile, and teeth whitening is a great option..."         |
| 100      | "Schwarz has been the president of the Greater Cleveland Film Commission for over..."       |
| 140      | "George Wales visits The Good Earth Chinese Restaurant in Wandsworth Common..."             |
| 180      | "Parenting Podcasts: You Aren't Alone! When you follow someone online..."                   |

**Topic distribution (200 samples):**

```
other (general)   94 (47%)
commerce          29 (14.5%)
tech              24 (12%)
sports            15 (7.5%)
health            11 (5.5%)
news              11 (5.5%)
cooking           10 (5%)
travel             6 (3%)
```

**Clustering analysis:**

- No runs of 5+ consecutive samples with same topic (excluding "other")
- Highly diverse content from first sample onward

### MathQA Analysis

MathQA is loaded entirely into memory from a ZIP file (~29K samples), not streamed. The in-memory list is shuffled using `random.shuffle()` after the train/eval split. No streaming bias concern applies.

---

## Revised Decision (2026-02-05)

**Do not implement shuffle buffer.**

### Rationale

1. **Empirical evidence contradicts the concern:** The "alphabetical trap" hypothesis predicted severe clustering (e.g., "first 10MB is all Django"). Investigation found Django files are 16.2% throughout with maximum cluster of 4 files. C4 showed no topic clustering at all.

2. **Diversity is already adequate:** Both streaming datasets show diverse content from the first samples onward. The data distribution supports our domain-level claims (code vs math vs prose specialization).

3. **Appropriate complexity:** For a small-scale project with <$100 GPU budget, adding complexity to solve a non-problem is not justified. The investigation itself provides the rigor needed for a technical report.

4. **Honest documentation:** Rather than blindly implementing a recommendation, we verified the underlying assumption and found it overstated.

---

## References

- Multi-model debate: `docs/models-debate/005a-DATA-PIPELINE-CRITICAL-REVIEW-OPUS-4-5.md` (Pass 3-11)
- Multi-model debate: `docs/models-debate/005b-DATA-PIPELINE-CRITICAL-REVIEW-GPT-5-2.md` (Pass 1)
- Multi-model debate: `docs/models-debate/005c-DATA-PIPELINE-CRITICAL-REVIEW-GEMINI-3.md`
- HuggingFace datasets streaming shuffle: https://huggingface.co/docs/datasets/stream#shuffle
