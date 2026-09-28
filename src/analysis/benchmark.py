"""
Multi-model benchmark for "Percepção dos Brasileiros acerca da Democracia" (CESOP 04832).

Implements Phase 2 of the paper methodology:
  - Loads LLM simulation JSONL results for all 4 evaluated models.
  - Loads CESOP ground-truth distributions from gabarito_cesop.jsonl
    (or from the raw .SAV file when available).
  - Computes JSD and Cramér's V for questions P02, P03, and P04.
  - Outputs a comparative performance table (reproduces Table II of the paper).
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.jsd import (
    compute_cramers_v,
    compute_jsd,
    compute_llm_distribution,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Survey metadata
# ---------------------------------------------------------------------------

QUESTIONS = ["P02", "P03", "P04"]

# Valid option letters per question (NS/NR codes excluded from evaluation)
QUESTION_OPTIONS = {
    "P02": list("abcdefghijkl"),   # 12 substantive alternatives (a–l)
    "P03": list("abcdef"),         # 6 substantive alternatives (a–f)
    "P04": list("abc"),            # 3 substantive alternatives (a–c)
}

# SPSS value-to-letter mappings for each question
P02_CODE_TO_LETTER = {float(i): chr(ord("a") + i - 1) for i in range(1, 13)}
P02_CODE_TO_LETTER[99.0] = "m"  # NS/NR

P03_CODE_TO_LETTER = {float(i): chr(ord("a") + i - 1) for i in range(1, 7)}
P03_CODE_TO_LETTER[99.0] = "g"  # NS/NR

P04_CODE_TO_LETTER = {1.0: "a", 2.0: "b", 3.0: "c", 99.0: "d"}  # NS/NR


# ---------------------------------------------------------------------------
# Ground-truth distributions from gabarito_cesop.jsonl
# ---------------------------------------------------------------------------

def load_gabarito_distributions(gabarito_path: Path) -> Dict[str, Dict[str, float]]:
    """Compute CESOP empirical distributions from pre-processed gabarito_cesop.jsonl.

    Each record in the file contains lists of coded answers per respondent.
    Multi-choice questions (P02, P03) are scored as mention rates:
    count(mentions of option x) / total_respondents.
    Single-choice question (P04) is scored as proportion of respondents per option.

    Args:
        gabarito_path: Path to gabarito_cesop.jsonl.

    Returns:
        Dictionary mapping question ID to {option: proportion}.
    """
    records = []
    with open(gabarito_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    counts = {q: defaultdict(int) for q in QUESTIONS}

    for rec in records:
        for letter in rec.get("P02", []):
            if letter in QUESTION_OPTIONS["P02"]:
                counts["P02"][letter] += 1
        for letter in rec.get("P03", []):
            if letter in QUESTION_OPTIONS["P03"]:
                counts["P03"][letter] += 1
        p04 = rec.get("P04")
        if p04 and p04 in QUESTION_OPTIONS["P04"]:
            counts["P04"][p04] += 1

    distributions = {}
    for q in QUESTIONS:
        opts = QUESTION_OPTIONS[q]
        total = n if q in ("P02", "P03") else sum(counts[q].values())
        distributions[q] = {
            opt: counts[q].get(opt, 0) / total if total > 0 else 0.0
            for opt in opts
        }

    return distributions


# ---------------------------------------------------------------------------
# Ground-truth distributions from raw .SAV file (optional)
# ---------------------------------------------------------------------------

def load_sav_distributions(sav_path: Path) -> Dict[str, Dict[str, float]]:
    """Compute CESOP empirical distributions from the raw SPSS .SAV file.

    Requires: pip install pyreadstat

    Args:
        sav_path: Path to the raw 04832.SAV file.

    Returns:
        Dictionary mapping question ID to {option: proportion}.
    """
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "pyreadstat is required to load .SAV files. "
            "Install it with: uv add pyreadstat"
        )

    df, _ = pyreadstat.read_sav(str(sav_path), apply_value_formats=False)
    n = len(df)

    def multi_dist(cols, code_map, valid):
        c = defaultdict(int)
        for col in cols:
            for val in df[col].dropna():
                letter = code_map.get(val)
                if letter and letter in valid:
                    c[letter] += 1
        return {opt: c.get(opt, 0) / n for opt in valid}

    def single_dist(col, code_map, valid):
        total = len(df[col].dropna())
        c = df[col].dropna().value_counts().to_dict()
        return {letter: c.get(code, 0) / total
                for code, letter in code_map.items() if letter in valid}

    return {
        "P02": multi_dist(["P2_1", "P2_2", "P2_3"], P02_CODE_TO_LETTER, QUESTION_OPTIONS["P02"]),
        "P03": multi_dist(
            [f"P3_{i}" for i in range(1, 7)], P03_CODE_TO_LETTER, QUESTION_OPTIONS["P03"]
        ),
        "P04": single_dist("P4", P04_CODE_TO_LETTER, QUESTION_OPTIONS["P04"]),
    }


# ---------------------------------------------------------------------------
# JSONL result parser
# ---------------------------------------------------------------------------

def _extract_answer_llama(content: str) -> str:
    """Extract answer letter from LLaMA 3.2 3B raw text output."""
    m = re.match(r'^\s*["`]?([a-nA-N])["`]?\s*[\n,]?', content)
    if m:
        return m.group(1).lower()
    try:
        obj = json.loads(content)
        return str(obj.get("answer", "")).strip().lower()
    except Exception:
        return ""


def _extract_answer_json(content: str) -> str:
    """Extract answer letter from JSON-structured API output (Maritaca, OpenAI)."""
    try:
        obj = json.loads(content)
        return str(obj.get("answer", "")).strip().lower()
    except Exception:
        m = re.search(r'"answer"\s*:\s*"([a-nA-N])"', content)
        if m:
            return m.group(1).lower()
        return ""


def parse_jsonl(path: Path, model_name: str) -> List[Dict]:
    """Parse a batch JSONL result file into structured answer records.

    Args:
        path: Path to the batch result JSONL file.
        model_name: Model identifier (used to select the answer extractor).

    Returns:
        List of dicts with keys {custom_id, question, rep, answer}.
    """
    is_llama = "llama" in model_name.lower()
    records = []
    errors = 0
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
                custom_id = obj.get("custom_id", "")

                # Extract question (P02/P03/P04) and repetition index from custom_id
                m = re.search(r"_(P0[234])_rep(\d+)$", custom_id)
                if not m:
                    errors += 1
                    continue
                question = m.group(1)
                rep = m.group(2)

                choices = (
                    obj.get("response", {})
                    .get("body", {})
                    .get("choices", [])
                )
                if not choices:
                    errors += 1
                    continue

                content_str = choices[0]["message"]["content"]
                answer = (
                    _extract_answer_llama(content_str)
                    if is_llama
                    else _extract_answer_json(content_str)
                )

                if not answer:
                    errors += 1
                    continue

                records.append(
                    {"custom_id": custom_id, "question": question, "rep": rep, "answer": answer}
                )
            except Exception:
                errors += 1

    print(f"    {path.name}: {total_lines} lines | {len(records)} parsed | {errors} errors")
    return records


# ---------------------------------------------------------------------------
# Main benchmark pipeline
# ---------------------------------------------------------------------------

def run_benchmark(
    results_dir: Path,
    gabarito_path: Path,
    sav_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full multi-model distributional benchmark.

    Reproduces Tables I and II from the paper.

    Args:
        results_dir: Directory containing one JSONL result file per model.
        gabarito_path: Path to gabarito_cesop.jsonl (pre-processed ground truth).
        sav_path: Optional path to raw 04832.SAV; used instead of gabarito if provided.

    Returns:
        DataFrame with columns [Model, Question, JSD, CramersV, p_value].
    """
    # Load ground truth
    print("[1] Loading empirical ground-truth distributions...")
    if sav_path and sav_path.exists():
        print(f"    Using raw SAV: {sav_path}")
        real_distributions = load_sav_distributions(sav_path)
    else:
        print(f"    Using gabarito: {gabarito_path}")
        real_distributions = load_gabarito_distributions(gabarito_path)

    for q, dist in real_distributions.items():
        opts = QUESTION_OPTIONS[q]
        print(f"    {q}: {dict(zip(opts, [round(dist[o], 3) for o in opts]))}")

    # Discover model result files
    model_files = {
        "sabia-4":      results_dir / "batch_result-sabia-4.jsonl",
        "sabiazinho-4": results_dir / "batch_result_sabiazinho-4.jsonl",
        "gpt-5-mini":   results_dir / "batch_result_gpt-5-mini.jsonl",
        "llama-3.2-3b": results_dir / "batch-result-llama.jsonl",
    }

    # Load LLM records
    print("\n[2] Loading LLM simulation results...")
    llm_records = {}
    for model_name, path in model_files.items():
        if not path.exists():
            print(f"    WARNING: {path.name} not found — skipping {model_name}")
            continue
        print(f"    {model_name}:")
        llm_records[model_name] = parse_jsonl(path, model_name)

    # Compute metrics
    print("\n[3] Computing JSD and Cramér's V...")
    results_table = []

    for q in QUESTIONS:
        opts = QUESTION_OPTIONS[q]
        real_dist = real_distributions[q]
        p_real_raw = np.array([real_dist.get(o, 0.0) for o in opts])
        p_real = p_real_raw / p_real_raw.sum() if p_real_raw.sum() > 0 else p_real_raw

        print(f"\n  {q} — real distribution: {dict(zip(opts, p_real.round(3)))}")

        for model_name, records in llm_records.items():
            llm_dist, total, _ = compute_llm_distribution(records, q, opts)
            p_llm_raw = np.array([llm_dist.get(o, 0.0) for o in opts])
            p_llm = p_llm_raw / p_llm_raw.sum() if p_llm_raw.sum() > 0 else p_llm_raw

            jsd_val = compute_jsd(p_llm, p_real)
            cv_val, p_val = compute_cramers_v(p_llm, p_real, n=10_000)

            results_table.append(
                {
                    "Model": model_name,
                    "Question": q,
                    "N": total,
                    "JSD": round(jsd_val, 4),
                    "CramersV": round(cv_val, 4) if not np.isnan(cv_val) else None,
                    "p_value": f"{p_val:.4e}" if not np.isnan(p_val) else None,
                }
            )
            print(
                f"    [{model_name}] JSD={jsd_val:.4f} | "
                f"Cramér's V={cv_val:.4f} | n={total}"
            )

    # Summary table
    df = pd.DataFrame(results_table)
    if df.empty:
        print("No results to display.")
        return df

    print("\n" + "=" * 70)
    print("COMPARATIVE PERFORMANCE TABLE (Table II)")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n--- Mean JSD Ranking (lower = better) ---")
    ranking = df.groupby("Model")["JSD"].mean().sort_values()
    for model, val in ranking.items():
        print(f"  {model:<18} {val:.4f}")

    return df
