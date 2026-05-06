#!/usr/bin/env python3
"""CLI entry points for running individual causal discovery methods."""

from __future__ import annotations

import argparse
from pathlib import Path

from .problem_benchmark import OUTPUT_ROOT, REPORTS_ROOT, ensure_processed, run_dataassist, run_guide, run_kcrl, run_unified, summarize_problem


def _run_single(method: str, problem: str, output_dir: Path) -> int:
    processed_dir = ensure_processed(problem)
    method_output_dir = OUTPUT_ROOT / method / problem
    if method == "DataAssist":
        row = run_dataassist(problem, processed_dir, method_output_dir)
    elif method == "GUIDE":
        row = run_guide(problem, processed_dir, method_output_dir)
    elif method == "KCRL":
        row = run_kcrl(problem, processed_dir, method_output_dir)
    elif method == "UnifiedOneShot":
        row = run_unified(problem, processed_dir, method_output_dir)
    else:
        raise ValueError(f"Unknown method: {method}")
    summarize_problem(problem, [row], output_dir)
    print(f"Wrote summary for {problem} to {output_dir / f'{problem}_summary.csv'}")
    return 0


def _build_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {method} on a single problem.")
    parser.add_argument("--problem", required=True, help="Problem name, e.g. sachs or SmBFO.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_ROOT)
    return parser


def run_dataassist_main() -> int:
    args = _build_parser("DataAssist").parse_args()
    return _run_single("DataAssist", args.problem, args.output_dir)


def run_guide_main() -> int:
    args = _build_parser("GUIDE").parse_args()
    return _run_single("GUIDE", args.problem, args.output_dir)


def run_kcrl_main() -> int:
    args = _build_parser("KCRL").parse_args()
    return _run_single("KCRL", args.problem, args.output_dir)


def run_unified_main() -> int:
    args = _build_parser("UnifiedOneShot").parse_args()
    return _run_single("UnifiedOneShot", args.problem, args.output_dir)
