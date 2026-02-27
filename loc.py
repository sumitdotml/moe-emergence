#!/usr/bin/env python3
"""
Code size and hotspot summary focused on signal over noise.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import token
import tokenize
from typing import Any

INCLUDED_EXTENSIONS = "py,cc,cpp,cxx,h,hh,hpp,hxx,cu,cuh"
LOC_MODES = {"all", "python", "core"}
SKIP_TOKEN_TYPES = {
    token.ENDMARKER,
    token.NEWLINE,
    token.NL,
    token.INDENT,
    token.DEDENT,
    tokenize.COMMENT,
}


def run_scc(root: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    cmd = [
        "scc",
        f"--include-ext={INCLUDED_EXTENSIONS}",
        "--no-cocomo",
        "--format",
        "json2",
        "--by-file",
        ".",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "scc is not installed. Install it first (example: brew install scc)."
        ) from exc
    if result.returncode != 0:
        err = result.stderr.strip() or "failed to run scc"
        raise RuntimeError(err)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("failed to parse scc output") from exc

    totals = {
        "files": 0,
        "lines": 0,
        "code": 0,
        "comments": 0,
        "blanks": 0,
        "complexity": 0,
    }
    files: list[dict[str, Any]] = []

    for lang in payload.get("languageSummary", []):
        totals["files"] += int(lang.get("Count", 0))
        totals["lines"] += int(lang.get("Lines", 0))
        totals["code"] += int(lang.get("Code", 0))
        totals["comments"] += int(lang.get("Comment", 0))
        totals["blanks"] += int(lang.get("Blank", 0))
        totals["complexity"] += int(lang.get("Complexity", 0))
        for file_entry in lang.get("Files", []):
            location = file_entry.get("Location")
            if not location:
                continue
            files.append(
                {
                    "path": Path(location),
                    "language": file_entry.get("Language", ""),
                    "lines": int(file_entry.get("Lines", 0)),
                    "code": int(file_entry.get("Code", 0)),
                    "comments": int(file_entry.get("Comment", 0)),
                    "blanks": int(file_entry.get("Blank", 0)),
                    "complexity": int(file_entry.get("Complexity", 0)),
                }
            )
    return totals, files


def run_scc_loc(root: Path, mode: str) -> int:
    include_extensions = "py" if mode == "python" else INCLUDED_EXTENSIONS
    target_path = root / "moe_emergence" if mode == "core" else root

    cmd = [
        "scc",
        f"--include-ext={include_extensions}",
        "--no-cocomo",
        "--no-complexity",
        str(target_path),
    ]
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "scc is not installed. Install it first (example: brew install scc)."
        ) from exc
    return result.returncode


def is_docstring_token(tok: tokenize.TokenInfo, prev_type: int) -> bool:
    if tok.type != token.STRING:
        return False
    if tok.start[1] != 0 and prev_type != token.INDENT:
        return False
    return prev_type in {token.INDENT, token.NEWLINE, token.DEDENT}


def python_token_stats(root: Path, files: list[dict[str, Any]]) -> dict[str, float]:
    densities: list[float] = []
    total_tokens = 0
    total_code_lines = 0

    for entry in files:
        if entry["path"].suffix != ".py":
            continue
        full_path = root / entry["path"]
        try:
            source = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = full_path.read_text(encoding="utf-8", errors="ignore")

        prev_type = token.INDENT
        token_count = 0
        code_lines: set[int] = set()

        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in SKIP_TOKEN_TYPES:
                prev_type = tok.type
                continue
            if is_docstring_token(tok, prev_type):
                prev_type = tok.type
                continue
            if tok.type in {token.NAME, token.NUMBER, token.OP, token.STRING}:
                token_count += 1
            code_lines.add(tok.start[0])
            prev_type = tok.type

        if not code_lines:
            continue
        file_density = token_count / len(code_lines)
        densities.append(file_density)
        total_tokens += token_count
        total_code_lines += len(code_lines)

    if not densities:
        return {}

    sorted_densities = sorted(densities)
    p90_idx = max(int(len(sorted_densities) * 0.9) - 1, 0)

    return {
        "files": float(len(densities)),
        "tokens_total": float(total_tokens),
        "code_lines_total": float(total_code_lines),
        "tokens_per_line_mean": total_tokens / total_code_lines,
        "tokens_per_line_median": statistics.median(sorted_densities),
        "tokens_per_line_p90": sorted_densities[p90_idx],
        "tokens_per_line_max": max(sorted_densities),
    }


def top_files(
    files: list[dict[str, Any]], key: str, limit: int
) -> list[dict[str, Any]]:
    return sorted(files, key=lambda x: x[key], reverse=True)[:limit]


def serialize_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**entry, "path": str(entry["path"])} for entry in files]


def print_top(label: str, files: list[dict[str, Any]], field: str, limit: int) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for entry in top_files(files, field, limit):
        print(f"{entry[field]:>6}  {entry['path']}")


def print_summary(
    totals: dict[str, int],
    files: list[dict[str, Any]],
    token_stats: dict[str, float],
    top_n: int,
) -> None:
    lines = totals["lines"] or 1
    code = totals["code"] or 1

    print("Code Size Summary")
    print("-----------------")
    print(f"Files:                  {totals['files']:,}")
    print(f"Lines (total):          {totals['lines']:,}")
    print(f"Code lines:             {totals['code']:,}")
    print(f"Comment lines:          {totals['comments']:,}")
    print(f"Blank lines:            {totals['blanks']:,}")
    print(f"Code ratio:             {totals['code'] / lines:.1%}")
    print(f"Comment/code ratio:     {totals['comments'] / code:.1%}")
    print(f"Complexity (total):     {totals['complexity']:,}")
    print(f"Complexity / 1k code:   {totals['complexity'] * 1000 / code:.1f}")

    if token_stats:
        print("\nPython Token Density")
        print("--------------------")
        token_rows = [
            ("Python files analyzed", f"{int(token_stats['files'])}"),
            ("Tokens (total)", f"{int(token_stats['tokens_total']):,}"),
            (
                "Overall mean (tokens/code line)",
                f"{token_stats['tokens_per_line_mean']:.2f}",
            ),
            (
                "File median (tokens/code line)",
                f"{token_stats['tokens_per_line_median']:.2f}",
            ),
            ("File p90 (tokens/code line)", f"{token_stats['tokens_per_line_p90']:.2f}"),
            ("File max (tokens/code line)", f"{token_stats['tokens_per_line_max']:.2f}"),
        ]
        token_label_width = max(len(label) for label, _ in token_rows)
        for label, value in token_rows:
            print(f"{label + ':':<{token_label_width + 2}} {value}")

    print_top(f"Top {top_n} by code lines", files, "code", top_n)
    print_top(f"Top {top_n} by complexity", files, "complexity", top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code size and hotspot summary")
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="repository root (default: repo root)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="number of files to show in each top list",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON summary",
    )
    parser.add_argument(
        "--loc",
        choices=sorted(LOC_MODES),
        help="print concise scc LOC table (all|python|core)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not (root / ".git").exists():
        print(f"not a repo root: {root}", file=sys.stderr)
        return 2

    if args.loc:
        try:
            return run_scc_loc(root, args.loc)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    try:
        totals, files = run_scc(root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    token_stats = python_token_stats(root, files)

    if args.json:
        payload = {
            "totals": totals,
            "token_stats": token_stats,
            "top_by_code": serialize_files(top_files(files, "code", args.top)),
            "top_by_complexity": serialize_files(
                top_files(files, "complexity", args.top)
            ),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print_summary(totals, files, token_stats, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
