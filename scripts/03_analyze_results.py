"""
Script 03: Compute JSD Metrics and Generate Comparative Results Table.

Evaluates the distributional alignment between LLM simulation outputs and
the empirical CESOP ground truth for questions P02, P03, and P04, using
Jensen-Shannon Divergence (JSD) and Cramér's V.

Reproduces Table II from the paper:
  "Comparative performance of models across questions using JSD."

By default, uses --gabarito-only mode (gabarito_cesop.jsonl), so no raw
.SAV file is required to reproduce the paper's results.

Usage:
    # Gabarito-only mode (default — no .SAV required):
    uv run python scripts/03_analyze_results.py \\
        --results-dir data/results/percepcao_democracia \\
        --gabarito data/cesop/gabarito_cesop.jsonl

    # With raw CESOP .SAV file (when available):
    uv run python scripts/03_analyze_results.py \\
        --results-dir data/results/percepcao_democracia \\
        --gabarito data/cesop/gabarito_cesop.jsonl \\
        --sav data/raw/04832.SAV
"""

import argparse
from pathlib import Path

from src.analysis.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute JSD distributional metrics for LLM silicon sampling results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results/percepcao_democracia",
        help="Directory containing JSONL result files per model.",
    )
    parser.add_argument(
        "--gabarito",
        type=str,
        default="data/cesop/gabarito_cesop.jsonl",
        help="Path to pre-processed gabarito_cesop.jsonl (derived CESOP ground truth).",
    )
    parser.add_argument(
        "--sav",
        type=str,
        default=None,
        help=(
            "Optional path to raw 04832.SAV file. If provided, distributions are "
            "computed from the original SPSS file instead of gabarito_cesop.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/percepcao_democracia/jsd_results.csv",
        help="Output CSV file for the comparative results table.",
    )
    parser.add_argument(
        "--gabarito-only",
        action="store_true",
        default=True,
        help="Use gabarito_cesop.jsonl only (default; ignores --sav if also provided).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    gabarito_path = Path(args.gabarito)
    sav_path = None if args.gabarito_only else (Path(args.sav) if args.sav else None)
    output_path = Path(args.output)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    if not gabarito_path.exists():
        print(f"ERROR: Gabarito file not found: {gabarito_path}")
        return

    df = run_benchmark(
        results_dir=results_dir,
        gabarito_path=gabarito_path,
        sav_path=sav_path,
    )

    if not df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
