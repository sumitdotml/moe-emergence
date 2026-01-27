⏺ Write(docs/decisions/005-phase3-data-sizing.md)
⎿  Wrote 150 lines to docs/decisions/005-phase3-data-sizing.md # Decision: Phase 3 Dataset Sizing and Token-Based Balancing
**Date:** 2025-12-26
**Status:** Accepted
**Context Commit:** `07c29ba`
--- ## Context
… +140 lines (ctrl+o to expand)

⏺ Now implementing data.py:

⏺ Write(moe_emergence/data.py)
⎿  Wrote 503 lines to moe_emergence/data.py
"""
Dataset Preparation for MoE Training
This module provides sequence packing and multi-domain dataset support for
training MoE models on code, math, and prose data.
Key features: - Sequence packing (no padding, efficient training) - Token-based sizing with imbalance warnings - Optional token balancing via truncation
… +493 lines (ctrl+o to expand)

⏺ Bash(uv run python moe_emergence/data.py --size-mb 0.5 --block-size 256) timeout: 5m 0s
⎿  Error: Exit code 1
Traceback (most recent call last):
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 502, in <module>
main()
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 409, in main
code_texts, code_info = load_code_data(
^^^^^^^^^^^^^^^
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 97, in
load_code_data
from datasets import load_dataset
ModuleNotFoundError: No module named 'datasets'
… +8 lines (ctrl+o to see all)

⏺ Need to add datasets dependency:

⏺ Bash(uv add datasets)
⎿  Resolved 73 packages in 326ms
Downloading pandas (10.2MiB)
Downloading pyarrow (32.6MiB)
… +29 lines (ctrl+o to expand)

⏺ Bash(uv run python moe_emergence/data.py --size-mb 0.5 --block-size 256) timeout: 5m 0s
⎿  Error: Exit code 1
Generating train split: 0%| | 0/7473 [00:00<?, ? examples/s]
split: 100%|██████████| 7473/7473 [00:00<00:00, 558518.06 examples/s]
Generating test split: 0%| | 0/1319 [00:00<?, ? examples/s]
100%|██████████| 1319/1319 [00:00<00:00, 519854.07 examples/s]
Traceback (most recent call last):
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 502, in <module>
main()
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 425, in main
prose_texts, prose_info = load_prose_data(
^^^^^^^^^^^^^^^^
File "/Users/sumit/playground/moe_emergence/moe_emergence/data.py", line 207, in
load_prose_data
… +28 lines (ctrl+o to see all)
⎿  Found 5 new diagnostic issues in 1 file (ctrl+o to expand)

⏺ OpenWebText has a compatibility issue. Let me use WikiText as an alternative and fix the linting warnings:

⏺ Update(moe_emergence/data.py)
⎿  Updated moe_emergence/data.py with 1 removal
25 from typing import Optional
26
27 import torch
28 - from torch import Tensor
28 from torch.utils.data import Dataset
29 from transformers import GPT2TokenizerFast
30

⏺ Update(moe_emergence/data.py)
⎿  Updated moe_emergence/data.py with 7 additions and 4 removals
191 min_example_chars: int = 200,
192 ) -> tuple[list[str], dict]:
193 """
194 - Load prose text from OpenWebText.
194 + Load prose text from WikiText-103.
195
196 Args:
197 max_size_mb: Target size in MB (character-based approximation)
...
203 """
204 from datasets import load_dataset
205
206 - ds = load_dataset("openwebtext", split="train", streaming=True)
206 + # WikiText-103 is a good alternative to OpenWebText
207 + ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", + split="train")
208
209 texts = []
210 total_chars = 0
...
213 if total_chars >= max_chars:
214 break
215 text = sample["text"]
216 - if min_example_chars < len(text) < max_example_chars:
216 + # Skip empty lines and very short texts
217 + if text and min_example_chars < len(text) < max_example_chars:
218 texts.append(text)
219 total_chars += len(text)
220
221 info = {
222 - "dataset": "openwebtext",
222 + "dataset": "Salesforce/wikitext",
223 + "config": "wikitext-103-raw-v1",
224 "split": "train",
225 "num_examples": len(texts),
226 "total_chars": total_chars,

⏺ Update(moe_emergence/data.py)
⎿  Updated moe_emergence/data.py with 1 addition and 1 removal
331 random.shuffle(self.blocks)
332
333 # Report final stats
334 - print(f"\nDataset ready:")
334 + print("\nDataset ready:")
335 print(f" Total blocks: {len(self.blocks)}")
336 print(f" Block size: {block_size} tokens")
337 print(f" Code: {self.block_counts['code']} blocks
({self.token_counts['code']:,} tokens)")

⏺ Update(moe_emergence/data.py)
⎿  Updated moe_emergence/data.py with 1 addition and 1 removal
399 )
400 args = parser.parse_args()
401
402 - print(f"Loading tokenizer...")
402 + print("Loading tokenizer...")
403 tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
404
405 print(f"\n{'='\*60}")

⏺ Search(pattern: "print\(f"", path: "moe_emergence/data.py", output_mode: "content")
⎿  Found 30 lines (ctrl+o to expand)

⏺ The remaining warnings appear to be false positives - those f-strings do have placeholders. Let me run the test:

⏺ Bash(uv run python moe_emergence/data.py --size-mb 0.5 --block-size 256) timeout: 5m 0s
⎿  Loading tokenizer...

     ============================================================
     … +75 lines (ctrl+o to expand)

⏺ Test passes. Let me verify the --balance-tokens flag works:

⏺ Bash(uv run python moe_emergence/data.py --size-mb 0.5 --block-size 256 --balance-tokens)
⎿  Loading tokenizer...

     ============================================================
     … +64 lines (ctrl+o to expand)

⏺ Balancing works correctly. Now let me update the decision doc to note the prose dataset change:

⏺ Update(docs/decisions/005-phase3-data-sizing.md)
⎿  Updated docs/decisions/005-phase3-data-sizing.md with 1 addition and 1 removal
92 |----------|------------|-------|
93 | `load_code_data()` | `codeparrot/codeparrot-clean` | Filter: 100 < len <
10000 chars |
94 | `load_math_data()` | `gsm8k` (main, train) + `hendrycks/competition_math`
(train) | Combined; consistent format |
95 - | `load_prose_data()` | `openwebtext` | Filter: len > 200 chars |
95 + | `load_prose_data()` | `Salesforce/wikitext` (wikitext-103-raw-v1) + | Filter: len > 200 chars |
96
97 **Math formatting** (consistent across GSM8K and MATH):
98 ```

⏺ Update(README.md)
⎿  Updated README.md with 5 additions and 4 removals
27 - [x] Full integration verification with actual GPT-2 model (10/10 tests
passed)
28 - See: `docs/experiments/run-001-gpt2-integration-verification.md`
29
30 - ### Phase 3: Dataset Preparation
30 + ### Phase 3: Dataset Preparation [DONE]
31
32 - - [ ] Sequence packing (not padding)
33 - - [ ] Code/math/prose data collection
34 - - [ ] `PackedMixedDomainDataset` class
32 + - [x] Sequence packing (not padding)
33 + - [x] Code/math/prose data collection
34 + - [x] `PackedMixedDomainDataset` class
35 + - See: `docs/decisions/005-phase3-data-sizing.md`
36
37 ### Phase 4: Training Infrastructure
38

⏺ Update(README.md)
⎿  Updated README.md with 1 addition and 1 removal
82 | Aux Collection | ✅ | `gpt2_moe.py` |
`collect_aux_outputs()` with clean probs |
83 | Verification | ✅ | `verify_gpt2_integration.py` | 10
comprehensive tests |
84 | Inference Playground | ✅ | `gpt2_inference.py` | Supports
vanilla/MoE/checkpoints |
85 - | Sequence Packing | ⬜ - | `data.py` | Efficient dataset without padding - |
85 + | Sequence Packing | ✅ + | `data.py` | Efficient dataset without padding + |
86 | Training Loop | ⬜ | `train.py` | With aux
loss collection |
87 | Collapse Detection | ⬜ | — | Early
stopping for ablation |
88 | Visualization | ⬜ | — | Heatmaps,
entropy plots |

⏺ Bash(git status --short)
⎿   M .claude/settings.local.json
M README.md
M pyproject.toml
… +3 lines (ctrl+o to expand)

⏺ ---
Phase 3 Complete! 🎉

Files Created/Modified

| File                                     | Status                                        |
| ---------------------------------------- | --------------------------------------------- |
| moe_emergence/data.py                    | New - sequence packing + multi-domain dataset |
| docs/decisions/005-phase3-data-sizing.md | New - design decision doc                     |
| README.md                                | Updated - Phase 3 marked done                 |
| pyproject.toml / uv.lock                 | Updated - added datasets dependency           |

Test Results

✓ Block size: 256 tokens
✓ No padding tokens
✓ Domain distribution: code=440, math=440, prose=440 (balanced)
✓ Collate function works
✅ Dataset ready for training!

Key Features Implemented

- Sequence packing - no padding, efficient training
- Token-based reporting - authoritative sizing metric
- Imbalance warning (>1.5x ratio detected)
- Optional balancing via --balance-tokens truncation
- CLI flags: --size-mb, --block-size, --balance-tokens, --seed, --max-example-chars

Note: Dataset Change

Used WikiText-103 instead of OpenWebText (compatibility issue with newer datasets library).
