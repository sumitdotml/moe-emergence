"""
One-time W&B Router Metrics Export

Exports router entropy and expert fraction data from W&B to local CSV.
Needed because metrics.jsonl only has loss/lr/throughput & router metrics
were logged to W&B only via tracking.py.

Usage:
    uv run python scripts/export_wandb.py
    uv run python scripts/export_wandb.py --runs moe-main  # single run
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).parent.parent
EXPORT_DIR = REPO_ROOT / ".cache" / "wandb_exports"

RUNS = {
    "moe-main": "sumit-ml/moe-emergence/j08s2d1m",
    "top-2": "sumit-ml/moe-emergence/6mw6qbac",
    "no-lb": "sumit-ml/moe-emergence/06pljhrv",
}

MOE_LAYERS = [8, 9, 10, 11]
N_EXPERTS = 8


def build_column_list() -> list[str]:
    cols = ["_step"]
    for layer in MOE_LAYERS:
        cols.append(f"router/layer_{layer}_entropy_per_token")
    for layer in MOE_LAYERS:
        for expert in range(N_EXPERTS):
            cols.append(f"router/layer_{layer}_expert_{expert}_frac")
    cols.extend(["router/entropy_per_token", "router/util_std"])
    return cols


def export_run(run_name: str, run_path: str) -> Optional[Path]:
    api = wandb.Api()
    run = api.run(run_path)

    columns = build_column_list()
    print(f"  Fetching {len(columns)} columns from {run_path}...")

    history = run.history(keys=columns[1:], pandas=True)

    if history.empty:
        print(f"  WARNING: No router metrics found for {run_name}")
        return None

    available = [c for c in columns if c in history.columns or c == "_step"]
    missing = [c for c in columns if c not in history.columns and c != "_step"]
    if missing:
        print(f"  Missing columns ({len(missing)}): {missing[:5]}...")

    router_cols = [c for c in available if c.startswith("router/")]
    history = history.dropna(subset=router_cols, how="all")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{run_name}-router-metrics.csv"
    history.to_csv(out_path, index=False)
    print(f"  Saved {len(history)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export W&B router metrics to CSV")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUNS.keys()),
        default=list(RUNS.keys()),
        help="Which runs to export (default: all)",
    )
    args = parser.parse_args()

    for run_name in args.runs:
        print(f"\nExporting {run_name}...")
        path = export_run(run_name, RUNS[run_name])
        if path is not None and path.is_file():
            df = pd.read_csv(path)
            print(f"  Verification: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
