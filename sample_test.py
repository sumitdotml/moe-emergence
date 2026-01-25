"""
Dataset sample testing script.

Run this BEFORE implementing data pipeline changes to verify dataset quality
and make informed decisions about code/prose dataset choices.

Usage: uv run python sample_test.py
"""

import random

from datasets import load_dataset

print("=" * 80)
print("DATASET SAMPLE TESTING - ALL THREE DOMAINS")
print("=" * 80)


# ::: CODE OPTIONS :::

print("\n" + "=" * 80)
print("### CODE Option 1: CodeParrot-clean ###")
print("=" * 80 + "\n")
code_ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
for i, sample in enumerate(code_ds):
    if i >= 5:
        break
    print(
        f"--- Sample {i + 1} | Repo: {sample['repo_name']} | License: {sample['license']} ---"
    )
    print(sample["content"][:1000])
    print("\n" + "-" * 40 + "\n")

# NOTE: StarCoderData requires TOS acceptance
# If accepted, I shall uncomment this:
# print("\n### CODE Option 2: StarCoderData ###\n")
# star_ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
# for i, sample in enumerate(star_ds):
#     if i >= 5: break
#     print(f"--- StarCoder Sample {i+1} ---")
#     print(sample['content'][:1000])
#     print()

# ::: MATH (MathQA) :::

print("\n" + "=" * 80)
print("### MATH: MathQA (DECIDED) ###")
print("=" * 80 + "\n")
math_ds = load_dataset("allenai/math_qa", split="train")
for i in random.sample(range(len(math_ds)), 5):
    sample = math_ds[i]
    print("--- Math Sample ---")
    print(f"Problem: {sample['Problem']}")
    print(f"Rationale: {sample['Rationale'][:500]}")
    print("\n" + "-" * 40 + "\n")


# ::: PROSE OPTIONS :::

print("\n" + "=" * 80)
print("### PROSE Option 1: WikiText-103 ###")
print("=" * 80 + "\n")
wiki_ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
count = 0
for i in random.sample(range(len(wiki_ds)), 20):
    sample = wiki_ds[i]
    if len(sample["text"]) > 100:  # skipping empty rows
        print("--- WikiText Sample ---")
        print(sample["text"][:800])
        print("\n" + "-" * 40 + "\n")
        count += 1
        if count >= 5:
            break

print("\n" + "=" * 80)
print("### PROSE Option 2: OpenWebText ###")
print("=" * 80 + "\n")
owt_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
for i, sample in enumerate(owt_ds):
    if i >= 5:
        break
    print(f"--- OpenWebText Sample {i + 1} ---")
    print(sample["text"][:800])
    print("\n" + "-" * 40 + "\n")

# ::: SUMMARY :::

print("\n" + "=" * 80)
print("DECISIONS NEEDED:")
print("=" * 80)
print(
    """
1. CODE: Is CodeParrot-clean quality good enough?
   - Should look for: Diverse styles, real production code, not repetitive
   - If not: must consider StarCoderData (requires TOS acceptance)

2. PROSE: Which is better for 3-domain separation?
   - WikiText-103: Encyclopedia articles only (cleaner but narrow)
   - OpenWebText: Diverse web content (journalism, blogs, opinions)

3. MATH: MathQA already decided (29K natural language word problems)
"""
)
