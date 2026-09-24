"""A1/A2/A3 domain quality summaries on the frozen A1 reference scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _cdf(values: pd.Series, reference: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    ref = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy())
    rank = np.searchsorted(ref, x.to_numpy(dtype=float), side="right")
    result = (rank + 0.5) / (len(ref) + 1.0)
    result[~np.isfinite(x.to_numpy(dtype=float))] = np.nan
    return pd.Series(result, index=x.index).clip(1e-9, 1.0)


def _score_with_references(raw: pd.DataFrame, refs: dict[str, pd.Series],
                           group_fields: dict[str, list[tuple[str, int]]]) -> pd.DataFrame:
    frame = raw.copy()
    for group, fields in group_fields.items():
        cols = []
        for field, direction in fields:
            source = -frame[field] if direction < 0 else frame[field]
            frame[field + "__q"] = _cdf(source, refs[field])
            cols.append(field + "__q")
        frame[group] = frame[cols].mean(axis=1, skipna=True)
    frame["Q_equal"] = frame[list(group_fields)].mean(axis=1, skipna=True).clip(1e-9, 1.0)
    return frame


def _weighted_score(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    values = frame[list(weights.index)]
    numerator = values.mul(weights, axis=1).sum(axis=1, min_count=1)
    denominator = values.notna().mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
    return (numerator / denominator).clip(1e-9, 1.0)


def _bootstrap_mean(x: np.ndarray, weights: np.ndarray | None = None,
                    n_boot: int = 2000, seed: int = 20260923) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[valid]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed + len(x))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        draws[i] = np.average(x[idx], weights=weights[idx]) if weights is not None and weights[idx].sum() > 0 else x[idx].mean()
    return tuple(np.quantile(draws, [0.025, 0.975]))


def run_quality_extensions(a_dir: Path, a1: pd.DataFrame, a1_raw: pd.DataFrame,
                           refs: dict[str, pd.Series], group_fields: dict[str, list[tuple[str, int]]],
                           critic: pd.Series, entropy: pd.Series,
                           quality_raw_frame, out: Path) -> pd.DataFrame:
    """Score full and non-overlap A2/A3 using A1's frozen empirical CDF references."""
    populations: list[tuple[str, pd.DataFrame]] = []
    base = a1.copy()
    base["Q_critic"] = _weighted_score(base, critic)
    base["Q_entropy"] = _weighted_score(base, entropy)
    populations.append(("A1", base))
    specs = [
        ("A2", "slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
        ("A3", "slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
    ]
    for label, filename, domain in specs:
        raw = quality_raw_frame(a_dir / filename, fallback=domain)
        frame = _score_with_references(raw, refs, group_fields)
        frame["Q_critic"] = _weighted_score(frame, critic)
        frame["Q_entropy"] = _weighted_score(frame, entropy)
        populations.append((label + "_full", frame))
        seen = set(a1_raw.loc[a1_raw.domain == domain, "id"].astype(str))
        populations.append((label + "_nonoverlap", frame.loc[~frame.id.astype(str).isin(seen)].copy()))

    rows = []
    for dataset, frame in populations:
        for domain, group in frame.groupby("domain", dropna=False):
            length = pd.to_numeric(group.rps_doc_word_count, errors="coerce").clip(lower=0).fillna(0).to_numpy()
            for weight, column in [("equal", "Q_equal"), ("critic", "Q_critic"),
                                   ("entropy", "Q_entropy"), ("length", "Q_equal")]:
                scores = pd.to_numeric(group[column], errors="coerce").to_numpy()
                use_weights = length if weight == "length" and length.sum() > 0 else None
                mean = np.average(scores[np.isfinite(scores)], weights=use_weights[np.isfinite(scores)]) if use_weights is not None else np.nanmean(scores)
                low, high = _bootstrap_mean(scores, use_weights)
                rows.append({"dataset": dataset, "domain": domain, "n": len(group), "weighting": weight,
                             "Q_mean": mean, "Q_CI_low": low, "Q_CI_high": high})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "q_domain_summary_A1_A2_A3_all.csv", index=False)

    ranks = summary.copy()
    ranks["rank"] = ranks.groupby(["dataset", "weighting"]).Q_mean.rank(ascending=False, method="average")
    ranks.to_csv(out / "q_domain_ranks_by_weighting.csv", index=False)
    stability = []
    for dataset, part in ranks.groupby("dataset"):
        wide = part.pivot(index="domain", columns="weighting", values="rank")
        for variant in ("critic", "entropy", "length"):
            pair = wide[["equal", variant]].dropna()
            stability.append({"dataset": dataset, "comparison": f"{variant}_vs_equal",
                              "n_domains": len(pair),
                              "spearman_rank": spearmanr(pair.equal, pair[variant]).statistic if len(pair) > 1 else np.nan,
                              "max_absolute_rank_shift": (pair.equal-pair[variant]).abs().max() if len(pair) else np.nan,
                              "domain_order_equal": bool((pair.equal == pair[variant]).all()) if len(pair) else False})
    pd.DataFrame(stability).to_csv(out / "q_domain_rank_stability.csv", index=False)

    compare = []
    for domain in ("arxiv", "github"):
        for dataset in ("A2_full", "A2_nonoverlap", "A3_full", "A3_nonoverlap"):
            if dataset.startswith("A2") and domain != "arxiv" or dataset.startswith("A3") and domain != "github":
                continue
            for weighting in ("equal", "critic", "entropy", "length"):
                a = summary[(summary.dataset == "A1") & (summary.domain == domain) & (summary.weighting == weighting)]
                b = summary[(summary.dataset == dataset) & (summary.domain == domain) & (summary.weighting == weighting)]
                if len(a) and len(b):
                    compare.append({"domain": domain, "extension": dataset, "weighting": weighting,
                                    "A1_n": int(a.n.iloc[0]), "extension_n": int(b.n.iloc[0]),
                                    "A1_Q_mean": a.Q_mean.iloc[0], "extension_Q_mean": b.Q_mean.iloc[0],
                                    "difference_extension_minus_A1": b.Q_mean.iloc[0]-a.Q_mean.iloc[0]})
    pd.DataFrame(compare).to_csv(out / "q_A1_A2_A3_domain_comparison.csv", index=False)
    return summary
