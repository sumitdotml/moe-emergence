"""
Phase 5 Analysis: Post-Training Computation

All analysis functions for the MoE emergence project. Handles model loading,
routing data collection, domain analysis, and token-type analysis.

Reuses existing functions from the codebase:
- gpt2_moe.install_moe_layers() / collect_aux_outputs()
- data.load_code_data() / load_math_data() / load_prose_data()
- data.split_texts_for_eval() / pack_sequences() / collate_packed()
"""

from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import keyword
from pathlib import Path
import tokenize

import pandas as pd
from safetensors.torch import load_file as load_safetensors
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizerBase

from moe_emergence.data import (
    collate_packed,
    load_code_data,
    load_math_data,
    load_prose_data,
    pack_sequences,
    split_texts_for_eval,
)
from moe_emergence.gpt2_moe import collect_aux_outputs, install_moe_layers

REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
HF_CACHE = CACHE_DIR / "huggingface"


type PackedBlock = dict[str, torch.Tensor | str]


class PackedBlocksDataset(Dataset[PackedBlock]):
    """Lightweight Dataset wrapper for packed block dicts."""

    def __init__(self, blocks: list[PackedBlock]) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> PackedBlock:
        return self._blocks[index]


def load_moe_model(
    checkpoint_stem: str, device: str = "cpu"
) -> tuple[GPT2LMHeadModel, dict, dict]:
    """
    Load a model from a checkpoint stem (expects .safetensors + .json sidecar).

    Args:
        checkpoint_stem: Path stem, e.g. "checkpoints/moe-main/final-model"
        device: Target device

    Returns:
        (model, moe_modules, metadata) — moe_modules is empty dict for dense
    """
    stem = Path(checkpoint_stem)
    meta_path = stem.with_suffix(".json")
    st_path = stem.with_suffix(".safetensors")

    with open(meta_path) as f:
        metadata = json.load(f)

    config = metadata.get("config", {})
    mode = metadata.get("mode", config.get("mode", "dense"))

    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=str(HF_CACHE))

    moe_modules = {}
    if mode == "moe":
        moe_layers = config.get("moe_layers", [8, 9, 10, 11])
        n_experts = config.get("n_experts", 8)
        topk = config.get("topk", 1)
        model, moe_modules = install_moe_layers(
            model, moe_layers=moe_layers, n_experts=n_experts, topk=topk
        )

    state_dict = load_safetensors(str(st_path), device="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, moe_modules, metadata


def build_domain_eval_blocks(
    domain: str,
    tokenizer: PreTrainedTokenizerBase,
    size_mb: float = 10.0,
    block_size: int = 512,
    seed: int = 42,
) -> list[PackedBlock]:
    """
    Build packed eval blocks for a single domain.

    Loads data, splits for eval (same seed=42 as training), packs sequences.
    Writes a hash manifest for reproducibility verification.
    """
    loaders = {
        "code": load_code_data,
        "math": load_math_data,
        "prose": load_prose_data,
    }
    loader = loaders[domain]

    texts, _ = loader(max_size_mb=size_mb)

    _, eval_texts = split_texts_for_eval(texts, seed)

    concat = "".join(eval_texts)
    sha = hashlib.sha256(concat.encode()).hexdigest()
    manifest_dir = CACHE_DIR / "eval_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{domain}.json"

    manifest = {
        "domain": domain,
        "n_texts": len(eval_texts),
        "n_chars": len(concat),
        "sha256": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "size_mb": size_mb,
        "seed": seed,
    }

    if manifest_path.exists():
        with open(manifest_path) as f:
            old = json.load(f)
        if old.get("sha256") != sha:
            print(
                f"WARNING: {domain} eval hash changed! "
                f"Old={old['sha256'][:12]}... New={sha[:12]}..."
            )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    blocks, _ = pack_sequences(eval_texts, tokenizer, block_size, domain_label=domain)
    return blocks


def compute_domain_expert_fractions(
    model: GPT2LMHeadModel,
    moe_modules: dict,
    domain_blocks: dict[str, list[PackedBlock]],
    device: str,
    batch_size: int = 4,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Compute per-domain expert utilization fractions.

    Args:
        model: Loaded model in eval mode
        moe_modules: dict[layer_idx, MoEWrapper]
        domain_blocks: {"code": [...], "math": [...], "prose": [...]}
        device: Device string
        batch_size: Batch size for forward passes

    Returns:
        {layer_idx: {"code": Tensor[n_experts], "math": ..., "prose": ...}}
    """
    n_experts = next(iter(moe_modules.values())).n_experts
    layer_indices = sorted(moe_modules.keys())

    results = {li: {} for li in layer_indices}

    for domain, blocks in domain_blocks.items():
        counts = {li: torch.zeros(n_experts) for li in layer_indices}

        loader = DataLoader(
            PackedBlocksDataset(blocks),
            batch_size=batch_size,
            collate_fn=collate_packed,
        )
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids)
            aux_list = collect_aux_outputs(moe_modules)
            for aux in aux_list:
                li = aux["layer_idx"]
                indices = aux["topk_indices"].cpu().view(-1)
                counts[li] += torch.bincount(indices, minlength=n_experts).float()

        for li in layer_indices:
            total = counts[li].sum()
            results[li][domain] = counts[li] / total if total > 0 else counts[li]

    return results


def compute_expert_utilization_at_snapshot(
    checkpoint_stem: str,
    all_blocks: list[PackedBlock],
    device: str,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """
    Load model from a snapshot and compute expert utilization across all blocks.

    Returns:
        {layer_idx: Tensor[n_experts]} — normalized fractions
    """
    model, moe_modules, _ = load_moe_model(checkpoint_stem, device)
    if not moe_modules:
        return {}

    n_experts = next(iter(moe_modules.values())).n_experts
    layer_indices = sorted(moe_modules.keys())
    counts = {li: torch.zeros(n_experts) for li in layer_indices}

    loader = DataLoader(
        PackedBlocksDataset(all_blocks),
        batch_size=batch_size,
        collate_fn=collate_packed,
    )
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)
        aux_list = collect_aux_outputs(moe_modules)
        for aux in aux_list:
            li = aux["layer_idx"]
            indices = aux["topk_indices"].cpu().view(-1)
            counts[li] += torch.bincount(indices, minlength=n_experts).float()

    results = {}
    for li in layer_indices:
        total = counts[li].sum()
        results[li] = counts[li] / total if total > 0 else counts[li]
    return results


def compute_collapse_trajectory(
    snapshot_stems: list[tuple[int, str]],
    all_blocks: list[PackedBlock],
    device: str,
    batch_size: int = 4,
) -> list[dict[int | str, torch.Tensor | int]]:
    """
    Compute expert utilization at each snapshot to show collapse over time.

    Args:
        snapshot_stems: [(step, "path/stem"), ...]
        all_blocks: Eval blocks to run through each snapshot
        device: Device string

    Returns:
        [{"step": 100, 8: Tensor[8], 9: Tensor[8], ...}, ...]
    """
    trajectory: list[dict[int | str, torch.Tensor | int]] = []
    for step, stem in snapshot_stems:
        print(f"  Computing utilization at step {step}...")
        util = compute_expert_utilization_at_snapshot(
            stem, all_blocks, device, batch_size
        )
        entry: dict[int | str, torch.Tensor | int] = {"step": step}
        for layer_idx, fracs in util.items():
            entry[layer_idx] = fracs
        trajectory.append(entry)
    return trajectory


def compute_token_type_routing(
    model: GPT2LMHeadModel,
    moe_modules: dict,
    code_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_samples: int = 200,
    max_seq_len: int = 512,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Analyze which experts handle which Python token types.

    Maps BPE tokens to Python token types (KEYWORD, NAME, NUMBER, STRING, OP,
    COMMENT, OTHER) via character offset overlap, then accumulates expert
    assignments per type.

    Returns:
        {layer_idx: {"KEYWORD": Tensor[n_experts], "NAME": ..., ...}}
    """
    n_experts = next(iter(moe_modules.values())).n_experts
    layer_indices = sorted(moe_modules.keys())

    token_types = ["KEYWORD", "NAME", "NUMBER", "STRING", "OP", "COMMENT", "OTHER"]
    counts = {
        li: {tt: torch.zeros(n_experts) for tt in token_types} for li in layer_indices
    }

    processed = 0
    for text in code_texts[:max_samples]:
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_seq_len,
        )
        offsets = enc["offset_mapping"][0]  # [seq_len, 2]
        input_ids = enc["input_ids"].to(device)

        try:
            py_tokens = list(tokenize.generate_tokens(StringIO(text).readline))
        except tokenize.TokenError:
            continue

        # converting line:col positions to char offsets for overlap with BPE spans
        line_offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                line_offsets.append(i + 1)

        char_types = ["OTHER"] * len(text)
        for tok in py_tokens:
            if tok.type == tokenize.ENCODING or tok.type == tokenize.ENDMARKER:
                continue
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            if start_line < 1 or start_line > len(line_offsets):
                continue

            start_char = line_offsets[start_line - 1] + start_col
            end_char = (
                line_offsets[min(end_line, len(line_offsets)) - 1] + end_col
                if end_line <= len(line_offsets)
                else len(text)
            )
            start_char = min(start_char, len(text))
            end_char = min(end_char, len(text))

            if tok.type == tokenize.NAME:
                tt = "KEYWORD" if keyword.iskeyword(tok.string) else "NAME"
            elif tok.type == tokenize.NUMBER:
                tt = "NUMBER"
            elif tok.type == tokenize.STRING:
                tt = "STRING"
            elif tok.type == tokenize.OP:
                tt = "OP"
            elif tok.type == tokenize.COMMENT:
                tt = "COMMENT"
            else:
                tt = "OTHER"

            for ci in range(start_char, end_char):
                char_types[ci] = tt

        with torch.no_grad():
            model(input_ids)
        aux_list = collect_aux_outputs(moe_modules)

        for aux in aux_list:
            li = aux["layer_idx"]
            topk_indices = aux["topk_indices"].cpu()  # [seq_len, topk]

            for bpe_idx in range(len(offsets)):
                start, end = offsets[bpe_idx].tolist()
                if start == end:
                    continue
                # using majority vote since a single BPE token can span multiple Python tokens
                span_types = char_types[start:end]
                type_counts = {}
                for t in span_types:
                    type_counts[t] = type_counts.get(t, 0) + 1
                majority_type = max(type_counts.items(), key=lambda kv: kv[1])[0]

                expert_id = topk_indices[bpe_idx, 0].item()
                counts[li][majority_type][expert_id] += 1

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{min(max_samples, len(code_texts))} files")

    results = {}
    for li in layer_indices:
        results[li] = {}
        for tt in token_types:
            total = counts[li][tt].sum()
            results[li][tt] = counts[li][tt] / total if total > 0 else counts[li][tt]

    return results


def load_metrics_jsonl(path: str | Path) -> pd.DataFrame:
    """Parse a metrics.jsonl file into a DataFrame."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_best_eval_metrics(metrics_path: str | Path) -> dict:
    """
    Get the best eval metrics from metrics.jsonl (row with lowest eval/loss).

    This is the correct source for "final eval" numbers across all runs,
    since some sidecar .json files lack eval metrics.
    """
    df = load_metrics_jsonl(metrics_path)
    eval_rows = df.dropna(subset=["eval/loss"])
    if eval_rows.empty:
        return {}
    best_idx = eval_rows["eval/loss"].idxmin()
    return eval_rows.loc[best_idx].dropna().to_dict()
