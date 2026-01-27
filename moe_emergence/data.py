"""
Dataset Preparation for MoE Training

This module provides sequence packing and multi-domain dataset support for
training MoE models on code, math, and prose data.

Key features:
- Sequence packing (no padding, efficient training)
- Token-based sizing with imbalance warnings
- Optional token balancing via truncation
- Reproducible shuffling with seed control

Usage:
    # Quick test
    uv run python moe_emergence/data.py --size-mb 1

    # Full dataset with balancing
    uv run python moe_emergence/data.py --size-mb 10 --balance-tokens

See docs/decisions/005-phase3-data-sizing.md and `docs/DATA-PIPELINE.md` for design decisions.
"""

import argparse
from pathlib import Path
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast

# all data cached in .cache/ at repository root
REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / ".cache"


def compute_eval_count(n_texts: int) -> int:
    """
    Computes number of texts to hold out for evaluation.

    Formula: min(max(10, int(n * 0.05)), int(n * 0.10))
    - At least 10 texts (for statistical reliability)
    - Target 5% of texts
    - Capped at 10% (to protect small domains)

    See docs/decisions/008-text-level-validation-split.md for rationale.
    """
    return min(max(10, int(n_texts * 0.05)), int(n_texts * 0.10))


def split_texts_for_eval(texts: list[str], seed: int) -> tuple[list[str], list[str]]:
    """
    Splits texts into train and eval sets at the text level.

    IMPORTANT: This must happen BEFORE packing to avoid document leakage.
    If we split after packing, the same document could span train and eval blocks.

    Args:
        texts: List of text strings
        seed: Random seed for reproducible shuffling

    Returns:
        Tuple of (train_texts, eval_texts) with no overlap
    """
    texts = list(texts)  # copy to avoid mutating input
    random.Random(seed).shuffle(texts)

    n_eval = compute_eval_count(len(texts))
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]

    return train_texts, eval_texts


def _dataset_meta(
    dataset, dataset_name: str, split: str, config: Optional[str] = None
) -> dict:
    info = dataset.info
    meta = {"dataset": dataset_name, "split": split}
    if config:
        meta["config"] = config
    if info is not None:
        if info.builder_name:
            meta["builder_name"] = info.builder_name
        if info.version is not None:
            meta["version"] = str(info.version)
    return meta


def pack_sequences(
    texts: list[str],
    tokenizer: GPT2TokenizerFast,
    block_size: int = 512,
    domain_label: Optional[str] = None,
) -> tuple[list[dict], int]:
    """
    Pack multiple texts into fixed-size blocks for efficient training.

    Instead of padding each text to max_length (wasteful), we:
    1. Tokenize all texts and concatenate with EOS separators
    2. Chunk into fixed-size blocks
    3. Each block may contain parts of multiple documents

    This is standard practice for causal LM pretraining.

    Args:
        texts: List of text strings
        tokenizer: GPT2TokenizerFast
        block_size: Size of each training block (default 512)
        domain_label: Domain label for all texts (e.g., 'code', 'math', 'prose')

    Returns:
        Tuple of (list of block dicts, total token count)
        Each block dict has 'input_ids' (Tensor) and 'domain' (str)
    """
    all_tokens = []

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    total_tokens = len(all_tokens)

    blocks = []
    for i in range(0, len(all_tokens) - block_size + 1, block_size):
        block_tokens = all_tokens[i : i + block_size]
        blocks.append(
            {
                "input_ids": torch.tensor(block_tokens, dtype=torch.long),
                "domain": domain_label or "unknown",
            }
        )

    return blocks, total_tokens


def load_code_data(
    max_size_mb: float = 10.0,
    max_example_chars: int = 10000,
    min_example_chars: int = 100,
) -> tuple[list[str], dict]:
    """
    Load Python code samples from CodeParrot.

    Args:
        max_size_mb: Target size in MB (character-based approximation)
        max_example_chars: Maximum characters per example
        min_example_chars: Minimum characters per example

    Returns:
        Tuple of (list of code texts, dataset info dict)
    """
    from datasets import load_dataset

    hf_cache = CACHE_DIR / "huggingface"
    ds = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
        cache_dir=str(hf_cache),
    )

    texts = []
    total_chars = 0
    max_chars = int(max_size_mb * 1024 * 1024)

    for sample in ds:
        if total_chars >= max_chars:
            break
        text = sample["content"]
        if min_example_chars < len(text) < max_example_chars:
            texts.append(text)
            total_chars += len(text)

    info = _dataset_meta(ds, "codeparrot/codeparrot-clean", "train")
    info.update(
        {
            "num_examples": len(texts),
            "total_chars": total_chars,
            "min_example_chars": min_example_chars,
            "max_example_chars": max_example_chars,
        }
    )

    return texts, info


def _download_mathqa() -> list[dict]:
    """
    Download MathQA dataset from the official source and cache locally.

    The HuggingFace loader for allenai/math_qa uses a deprecated script format,
    so we download directly from the source ZIP file.

    Returns:
        List of dicts with 'Problem' and 'Rationale' keys (and other fields)

    Example sample::

        {
            "Problem": "the banker's gain of a certain sum due 3 years hence
                        at 10% per annum is rs. 36. what is the present worth?",
            "Rationale": "explanation: t = 3 years r = 10% td = (bg × 100) / tr
                          = (36 × 100) / (3 × 10) = 12 × 10 = rs. 120 ...
                          answer: option a",
            "options": "a) rs. 400, b) rs. 300, c) rs. 500, ...",
            "correct": "a",
            "category": "gain"
        }

    Note: Rationales contain Unicode math symbols (×, ⇒, ⋅) mixed with ASCII.
    See docs/decisions/007-math-prefix-randomization.md for details.
    """
    import io
    import json
    import urllib.request
    import zipfile

    cache_dir = CACHE_DIR / "mathqa"
    cache_file = cache_dir / "train.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = "https://math-qa.github.io/math-QA/data/MathQA.zip"
    print(f"  Downloading MathQA from {url}...")

    with urllib.request.urlopen(url) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        with archive.open("train.json") as f:
            data = json.load(f)

    # caching for future runs
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f)

    print(f"  Cached to {cache_file}")
    return data


def load_math_data(
    max_size_mb: float = 10.0,
    max_example_chars: int = 10000,
) -> tuple[list[str], dict]:
    """
    Load math problems from MathQA (allenai).

    Format: "{Problem}\n\n{Rationale}" (no prefixes - see decision 007)

    The math content is naturally distinguishable through numbers, operators,
    step-by-step calculations, and mathematical vocabulary.

    Args:
        max_size_mb: Target size in MB (character-based approximation)
        max_example_chars: Maximum characters per example

    Returns:
        Tuple of (list of math texts, dataset info dict)
    """
    mathqa_samples = _download_mathqa()

    texts = []
    total_chars = 0
    max_chars = int(max_size_mb * 1024 * 1024)

    for sample in mathqa_samples:
        if total_chars >= max_chars:
            break

        # Format: problem text, blank line, rationale
        # No prefixes - content naturally signals "math" domain
        text = f"{sample['Problem']}\n\n{sample['Rationale']}"

        if len(text) < max_example_chars:
            texts.append(text)
            total_chars += len(text)

    info = {
        "dataset": "MathQA (allenai)",
        "source_url": "https://math-qa.github.io/math-QA/data/MathQA.zip",
        "total_available": len(mathqa_samples),
        "num_examples": len(texts),
        "total_chars": total_chars,
        "max_example_chars": max_example_chars,
    }

    return texts, info


def load_prose_data(
    max_size_mb: float = 10.0,
    max_example_chars: int = 10000,
    min_example_chars: int = 200,
) -> tuple[list[str], dict]:
    """
    Load prose text from WikiText-103.

    Args:
        max_size_mb: Target size in MB (character-based approximation)
        max_example_chars: Maximum characters per example
        min_example_chars: Minimum characters per example

    Returns:
        Tuple of (list of prose texts, dataset info dict)
    """
    from datasets import load_dataset

    hf_cache = CACHE_DIR / "huggingface"
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        cache_dir=str(hf_cache),
    )

    texts = []
    total_chars = 0
    max_chars = int(max_size_mb * 1024 * 1024)

    for sample in ds:
        if total_chars >= max_chars:
            break
        text = sample["text"]
        # skipping empty lines and very short texts
        if text and min_example_chars < len(text) < max_example_chars:
            texts.append(text)
            total_chars += len(text)

    info = _dataset_meta(ds, "Salesforce/wikitext", "train", "wikitext-103-raw-v1")
    info.update(
        {
            "num_examples": len(texts),
            "total_chars": total_chars,
            "min_example_chars": min_example_chars,
            "max_example_chars": max_example_chars,
        }
    )

    return texts, info


class PackedMixedDomainDataset(Dataset):
    """
    Dataset combining packed sequences from multiple domains.

    Maintains domain labels at block level for post-hoc analysis,
    but all blocks are shuffled together for training.

    Args:
        code_texts: List of code text samples
        math_texts: List of math text samples
        prose_texts: List of prose text samples
        tokenizer: GPT2TokenizerFast
        block_size: Tokens per block (default 512)
        balance_tokens: If True, truncate larger domains to match smallest
        seed: Random seed for reproducible shuffling
    """

    def __init__(
        self,
        code_texts: list[str],
        math_texts: list[str],
        prose_texts: list[str],
        tokenizer: GPT2TokenizerFast,
        block_size: int = 512,
        balance_tokens: bool = False,
        seed: int = 42,
    ) -> None:
        self.block_size = block_size

        # packing each domain separately
        print("Packing code sequences...")
        code_blocks, code_tokens = pack_sequences(
            code_texts, tokenizer, block_size, "code"
        )

        print("Packing math sequences...")
        math_blocks, math_tokens = pack_sequences(
            math_texts, tokenizer, block_size, "math"
        )

        print("Packing prose sequences...")
        prose_blocks, prose_tokens = pack_sequences(
            prose_texts, tokenizer, block_size, "prose"
        )

        # storing token counts for reporting
        self.token_counts = {
            "code": code_tokens,
            "math": math_tokens,
            "prose": prose_tokens,
        }

        self.block_counts = {
            "code": len(code_blocks),
            "math": len(math_blocks),
            "prose": len(prose_blocks),
        }

        # checking for imbalance
        counts = list(self.token_counts.values())
        min_tokens = min(counts)
        max_tokens = max(counts)

        if max_tokens > 1.5 * min_tokens:
            min_domain = min(self.token_counts, key=self.token_counts.get)
            print(
                f"\nWarning: Token imbalance detected (>1.5x ratio)!"
                f"\n  Smallest: {min_domain} with {min_tokens:,} tokens"
                f"\n  Largest has {max_tokens:,} tokens"
            )
            if not balance_tokens:
                print("  Use --balance-tokens to truncate to smallest domain.\n")

        # optional: balance by truncating to smallest domain
        if balance_tokens:
            min_blocks = min(len(code_blocks), len(math_blocks), len(prose_blocks))
            code_blocks = code_blocks[:min_blocks]
            math_blocks = math_blocks[:min_blocks]
            prose_blocks = prose_blocks[:min_blocks]

            total_blocks = min_blocks * 3
            total_tokens = min_blocks * block_size * 3
            print(
                f"\nBalanced to {min_blocks} blocks per domain "
                f"({total_blocks} total, ~{total_tokens:,} tokens)"
            )

            # updating counts after balancing
            self.block_counts = {
                "code": min_blocks,
                "math": min_blocks,
                "prose": min_blocks,
            }
            self.token_counts = {
                "code": min_blocks * block_size,
                "math": min_blocks * block_size,
                "prose": min_blocks * block_size,
            }

        # combining and shuffling
        self.blocks = code_blocks + math_blocks + prose_blocks
        random.seed(seed)
        random.shuffle(self.blocks)

        # reporting final stats
        print("\nDataset ready:")
        print(f"  Total blocks: {len(self.blocks)}")
        print(f"  Block size: {block_size} tokens")
        print(
            f"  Code: {self.block_counts['code']} blocks "
            f"({self.token_counts['code']:,} tokens)"
        )
        print(
            f"  Math: {self.block_counts['math']} blocks "
            f"({self.token_counts['math']:,} tokens)"
        )
        print(
            f"  Prose: {self.block_counts['prose']} blocks "
            f"({self.token_counts['prose']:,} tokens)"
        )

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict:
        block = self.blocks[idx]
        return {"input_ids": block["input_ids"], "domain": block["domain"]}


def collate_packed(batch: list[dict]) -> dict:
    """
    Collate function for packed sequences.

    No padding needed since all blocks are the same size.

    Args:
        batch: List of dicts with 'input_ids' and 'domain'

    Returns:
        Dict with stacked 'input_ids' tensor and 'domains' list
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "domains": [item["domain"] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare packed dataset for MoE training"
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=10.0,
        help="Target size per domain in MB (default: 10)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Tokens per block (default: 512)",
    )
    parser.add_argument(
        "--balance-tokens",
        action="store_true",
        help="Truncate larger domains to match smallest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    parser.add_argument(
        "--max-example-chars",
        type=int,
        default=10000,
        help="Per-example length cap (default: 10000)",
    )
    args = parser.parse_args()

    print("Loading tokenizer...")
    hf_cache = CACHE_DIR / "huggingface"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=str(hf_cache))

    print(f"\n{'=' * 60}")
    print(f"Loading datasets (target: {args.size_mb}MB per domain, pre-split)")
    print(f"{'=' * 60}\n")

    # loading all domains
    print("Loading code data...")
    code_texts, code_info = load_code_data(
        max_size_mb=args.size_mb,
        max_example_chars=args.max_example_chars,
    )
    print(
        f"  Loaded {code_info['num_examples']} examples "
        f"({code_info['total_chars']:,} chars)"
    )

    print("\nLoading math data...")
    math_texts, math_info = load_math_data(
        max_size_mb=args.size_mb,
        max_example_chars=args.max_example_chars,
    )
    print(
        f"  Loaded {math_info['num_examples']} examples "
        f"({math_info['total_chars']:,} chars) from MathQA"
    )

    print("\nLoading prose data...")
    prose_texts, prose_info = load_prose_data(
        max_size_mb=args.size_mb,
        max_example_chars=args.max_example_chars,
    )
    print(
        f"  Loaded {prose_info['num_examples']} examples "
        f"({prose_info['total_chars']:,} chars)"
    )

    # gotta split each domain into train/eval at text level before packing
    print(f"\n{'=' * 60}")
    print("Splitting texts into train/eval (text-level, before packing)")
    print(f"{'=' * 60}\n")

    code_train, code_eval = split_texts_for_eval(code_texts, args.seed)
    math_train, math_eval = split_texts_for_eval(math_texts, args.seed)
    prose_train, prose_eval = split_texts_for_eval(prose_texts, args.seed)

    print(f"Code:  {len(code_train):,} train / {len(code_eval):,} eval texts")
    print(f"Math:  {len(math_train):,} train / {len(math_eval):,} eval texts")
    print(f"Prose: {len(prose_train):,} train / {len(prose_eval):,} eval texts")

    # verifying for possible train/eval leakage
    assert len(set(code_train) & set(code_eval)) == 0, "Code train/eval overlap!"
    assert len(set(math_train) & set(math_eval)) == 0, "Math train/eval overlap!"
    assert len(set(prose_train) & set(prose_eval)) == 0, "Prose train/eval overlap!"
    print("\n[OK] No train/eval text overlap (leakage check passed)")

    print(f"\n{'=' * 60}")
    print("Creating packed TRAIN dataset")
    print(f"{'=' * 60}\n")

    train_dataset = PackedMixedDomainDataset(
        code_texts=code_train,
        math_texts=math_train,
        prose_texts=prose_train,
        tokenizer=tokenizer,
        block_size=args.block_size,
        balance_tokens=args.balance_tokens,
        seed=args.seed,
    )

    print(f"\n{'=' * 60}")
    print("Creating packed EVAL dataset")
    print(f"{'=' * 60}\n")

    eval_dataset = PackedMixedDomainDataset(
        code_texts=code_eval,
        math_texts=math_eval,
        prose_texts=prose_eval,
        tokenizer=tokenizer,
        block_size=args.block_size,
        balance_tokens=False,  # never balance eval
        seed=args.seed,
    )

    # verification
    print(f"\n{'=' * 60}")
    print("Verification")
    print(f"{'=' * 60}\n")

    # checking block sizes
    sample = train_dataset[0]
    assert sample["input_ids"].shape == (args.block_size,), (
        f"Block size mismatch: {sample['input_ids'].shape} != ({args.block_size},)"
    )
    print(f"[OK] Block size: {args.block_size} tokens")

    # checking no padding tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        print("[OK] Pad token id is None; skipping pad-token check")
    else:
        has_padding = any(
            (train_dataset[i]["input_ids"] == pad_token_id).any()
            for i in range(min(100, len(train_dataset)))
        )
        assert not has_padding, "Found padding tokens in dataset!"
        print("[OK] No padding tokens")

    # testing collate function
    from torch.utils.data import DataLoader

    loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_packed)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, args.block_size), (
        f"Batch shape mismatch: {batch['input_ids'].shape}"
    )
    assert len(batch["domains"]) == 4
    print("[OK] Collate function works")

    print(f"\n{'=' * 60}")
    print("Summary (for reproducibility)")
    print(f"{'=' * 60}\n")

    print(f"Seed: {args.seed}")
    print(f"Block size: {args.block_size}")
    print(f"Balance tokens (train): {args.balance_tokens}")
    print()
    print("Text counts (pre-packing):")
    print(f"  Code:  {len(code_train):,} train / {len(code_eval):,} eval")
    print(f"  Math:  {len(math_train):,} train / {len(math_eval):,} eval")
    print(f"  Prose: {len(prose_train):,} train / {len(prose_eval):,} eval")
    print()
    print("Block counts (post-packing):")
    print(f"  Train: {len(train_dataset):,} blocks")
    print(f"  Eval:  {len(eval_dataset):,} blocks")

    print("\n[OK] Datasets ready for training.")
    print(f"   Train blocks: {len(train_dataset)}")
    print(f"   Eval blocks:  {len(eval_dataset)}")
    print(f"   Example: ~{len(train_dataset) // 8} steps if batch_size=8, 1 epoch")


if __name__ == "__main__":
    main()
