"""Domain-standardized quality conflict definitions and summaries."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

GROUPS = ["edu", "read", "reason", "clean", "struct"]
THRESHOLDS = [(0.70, 0.30), (0.75, 0.25), (0.80, 0.20), (0.90, 0.10)]


def conflict_details(frame: pd.DataFrame, upper: float = 0.75, lower: float = 0.25):
    """Return row flags and unordered high-low group-pair counts within one domain."""
    ranks = frame[GROUPS].rank(pct=True, axis=0, method="average")
    high = ranks.ge(upper)
    low = ranks.le(lower)
    flags = (high.sum(axis=1).gt(0) & low.sum(axis=1).gt(0))
    pair_rows = []
    for a, b in combinations(GROUPS, 2):
        count = int(((high[a] & low[b]) | (high[b] & low[a])).sum())
        pair_rows.append({"group_a": a, "group_b": b, "count": count})
    pairs = pd.DataFrame(pair_rows)
    pairs["pair_rate_all"] = pairs["count"] / max(1, len(frame))
    pairs["share_of_conflict_pairs"] = pairs["count"] / max(1, pairs["count"].sum())
    return flags, pairs


def penalized_score(frame: pd.DataFrame, lam: float) -> pd.Series:
    return (frame["Q_equal"] - lam * conflict_severity_from_ranks(frame)).clip(1e-9, 1.0)


def conflict_severity_from_ranks(frame: pd.DataFrame) -> pd.Series:
    """Domain-relative disagreement: mean absolute distance from the median group rank."""
    ranks = frame[GROUPS].rank(pct=True, axis=0, method="average")
    return ranks.sub(ranks.median(axis=1), axis=0).abs().mean(axis=1).mul(2).clip(0, 1)


def summarize_conflicts(frame: pd.DataFrame, dataset: str):
    rates, pair_tables = [], []
    for domain, group in frame.groupby("domain", dropna=False):
        for upper, lower in THRESHOLDS:
            flags, pairs = conflict_details(group, upper, lower)
            rates.append({"dataset": dataset, "domain": domain, "n": len(group),
                          "upper_quantile": upper, "lower_quantile": lower,
                          "conflict_n": int(flags.sum()), "conflict_rate": float(flags.mean())})
            if upper == 0.75:
                pairs.insert(0, "domain", domain)
                pairs.insert(0, "dataset", dataset)
                pairs["upper_quantile"] = upper
                pairs["lower_quantile"] = lower
                pair_tables.append(pairs)
    return (pd.DataFrame(rates),
            pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame())
