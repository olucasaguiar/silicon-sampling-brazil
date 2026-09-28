"""
Analysis utilities for distributional evaluation of LLM silicon sampling.

Provides:
- compute_jsd: Jensen-Shannon Divergence (base-2) between two distributions.
- compute_cramers_v: Cramér's V association statistic between LLM and real distributions.
- normalize: Normalize a dictionary of counts to a probability vector.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence in base-2 between distributions p and q.

    JSD is bounded within [0, 1]:
    - JSD = 0 means identical distributions.
    - JSD < 0.15 is adopted as the threshold for close distributional convergence
      (following Miranda et al., 2025).
    - JSD > 0.50 indicates severe empirical divergence.

    Args:
        p: Simulated distribution (LLM responses), unnormalized.
        q: Empirical distribution (CESOP ground truth), unnormalized.

    Returns:
        JSD value in [0, 1].
    """
    eps = 1e-10
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    # jensenshannon returns sqrt(JSD); we square to get JSD
    return float(jensenshannon(p, q, base=2)) ** 2


def compute_cramers_v(
    p_llm: np.ndarray, p_real: np.ndarray, n: int = 2000
) -> tuple[float, float]:
    """Compute Cramér's V association between LLM and real distributions.

    Builds a 2×K contingency table from simulated counts (scaled by n) and
    applies chi-squared test to derive Cramér's V.

    Args:
        p_llm: Normalized LLM response distribution vector.
        p_real: Normalized real empirical distribution vector.
        n: Number of observations used to scale proportions into counts.

    Returns:
        Tuple of (cramers_v, p_value). Returns (nan, nan) on failure.
    """
    obs = np.round(p_llm * n).astype(int)
    exp = np.round(p_real * n).astype(int)
    obs = np.maximum(obs, 0)
    exp = np.maximum(exp, 0)

    table = np.array([obs, exp])
    col_sums = table.sum(axis=0)
    table = table[:, col_sums > 0]

    if table.shape[1] < 2:
        return np.nan, np.nan

    try:
        chi2, p_val, dof, _ = chi2_contingency(table)
        n_total = table.sum()
        k = min(table.shape)
        v = np.sqrt(chi2 / (n_total * (k - 1))) if (n_total * (k - 1)) > 0 else 0.0
        return float(v), float(p_val)
    except Exception:
        return np.nan, np.nan


def normalize_distribution(counts: Dict[str, int | float], keys: List[str]) -> np.ndarray:
    """Extract and normalize a probability vector from a counts dictionary.

    Args:
        counts: Dictionary mapping option keys to counts or proportions.
        keys: Ordered list of valid option keys to extract.

    Returns:
        Normalized numpy array of probabilities summing to 1.
    """
    v = np.array([counts.get(k, 0.0) for k in keys], dtype=float)
    s = v.sum()
    return v / s if s > 0 else v


def compute_llm_distribution(
    records: List[Dict],
    question: str,
    valid_options: List[str],
) -> tuple[Dict[str, float], int, Dict[str, int]]:
    """Compute the empirical response distribution from LLM JSONL batch results.

    Args:
        records: Parsed JSONL records with fields {question, rep, answer}.
        question: Question identifier (e.g., "P02", "P03", "P04").
        valid_options: List of valid option letters (e.g., ["a", "b", ..., "l"]).

    Returns:
        Tuple of (distribution_dict, total_count, raw_counts).
    """
    q_records = [r for r in records if r["question"] == question]
    total = len(q_records)
    counts: Dict[str, int] = defaultdict(int)
    for r in q_records:
        counts[r["answer"]] += 1

    dist: Dict[str, float] = {}
    for opt in valid_options:
        dist[opt] = counts.get(opt, 0) / total if total > 0 else 0.0

    # Track invalid/NS-NR responses separately for diagnostic purposes
    ns_nr = sum(v for k, v in counts.items() if k not in valid_options and k in "mn")
    invalid = sum(v for k, v in counts.items() if k not in valid_options and k not in "mn")
    dist["_ns_nr"] = ns_nr / total if total > 0 else 0.0
    dist["_invalid"] = invalid / total if total > 0 else 0.0

    return dist, total, dict(counts)
