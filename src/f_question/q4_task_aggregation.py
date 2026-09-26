"""Aggregate C8 task-level evaluations and estimate task-specific trends."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


TASKS = {
    "leaderboard_arc_challenge": "ARC-Challenge",
    "leaderboard_musr": "MuSR",
    "leaderboard_ifeval": "IFEval",
    "leaderboard_math_hard": "MATH-Hard",
    "leaderboard_gpqa": "GPQA",
    "leaderboard_bbh": "BBH",
    "leaderboard_mmlu_pro": "MMLU-Pro",
}
TASK_COLUMNS = {
    "IFEval": "IFEval", "BBH": "BBH", "MATH-Hard": "MATH_Lvl5",
    "GPQA": "GPQA", "MuSR": "MUSR", "MMLU-Pro": "MMLU_PRO",
}


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _score(task: str, value: dict[str, Any]) -> tuple[float, str] | None:
    preferred = ["prompt_level_strict_acc,none"] if task == "IFEval" else ["acc_norm,none"]
    fallback = ["acc,none", "exact_match,none", "prompt_level_strict_acc,none"]
    for metric in preferred + fallback:
        raw = value.get(metric)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            return score, metric
    return None


def aggregate_c8(folder: Path, leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return C8 model-task scores and a file-level parsing audit.

    Model parameters/dates are joined from C1 because the supplied C4 table
    does not share model identifiers with most C8 leaderboard submissions.
    """
    lb = leaderboard.copy()
    lb["_key"] = lb["Model"].map(_key)
    lb = lb.drop_duplicates("_key").set_index("_key")
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit.append({"path": str(path), "status": "malformed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        model = str(payload.get("model_name", ""))
        model_key = _key(model)
        matched = model_key in lb.index
        meta = lb.loc[model_key] if matched else None
        date = pd.to_datetime(payload.get("date"), unit="s", utc=True, errors="coerce")
        count = 0
        results = payload.get("results", {})
        groups = payload.get("groups", {})
        for group, task in TASKS.items():
            value = groups.get(group) or results.get(group)
            if not isinstance(value, dict):
                continue
            extracted = _score(task, value)
            if extracted is None:
                continue
            score, metric = extracted
            rows.append({
                "model": model, "date_utc": date, "year": date.year if not pd.isna(date) else np.nan,
                "task": task, "score": score * 100.0, "metric": metric,
                "params_b_c1": pd.to_numeric(meta.get("#Params (B)"), errors="coerce") if matched else np.nan,
                "c1_model_match": matched,
            })
            count += 1
        audit.append({"path": str(path), "status": "parsed", "model": model,
                      "c1_model_match": matched, "task_scores": count})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def c3_task_heterogeneity(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Fit score ~ log10(parameters) + C3 publication year for each task."""
    rows = []
    for task, col in TASK_COLUMNS.items():
        if col not in timeseries:
            continue
        d = timeseries[["Model", "Year", "Params_B", col]].copy()
        d["year"] = pd.to_numeric(d.Year, errors="coerce")
        d["params"] = pd.to_numeric(d.Params_B, errors="coerce")
        d["score"] = pd.to_numeric(d[col], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["year", "params", "score"])
        d = d[(d.params > 0) & (d.year >= 2022)]
        if len(d) < 5:
            continue
        X = np.column_stack([np.ones(len(d)), np.log10(d.params), d.year - d.year.min()])
        y = d.score.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        fit = stats.linregress(d.year, d.score)
        rows.append({"task": task, "c3_score_column": col, "n": len(d),
                     "year_start": int(d.year.min()), "year_end": int(d.year.max()),
                     "year_slope_adjusted_points_per_year": beta[2],
                     "scale_slope_points_per_log10_params": beta[1], "r2_adjusted": r2,
                     "year_slope_unadjusted": fit.slope, "year_slope_pvalue_unadjusted": fit.pvalue})
    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def c8_task_scale_heterogeneity(task_scores: pd.DataFrame) -> pd.DataFrame:
    """Fit C8 task score against model scale without using evaluation time.

    C8 ``date_utc`` is an evaluation timestamp rather than a publication date,
    so this is a cross-sectional model only.  The output remains separate from
    :func:`c3_task_heterogeneity` because their samples and score scales differ.
    """
    columns = ["task", "n", "scale_slope_points_per_log10_params", "r2",
               "spearman_rho", "params_coverage"]
    if task_scores.empty or not {"task", "score", "params_b_c1"}.issubset(task_scores.columns):
        return pd.DataFrame(columns=columns)
    d = task_scores.copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["params_b_c1"] = pd.to_numeric(d["params_b_c1"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for task, group in d.groupby("task", sort=True):
        valid = group[group["score"].notna() & group["params_b_c1"].gt(0)].copy()
        if len(valid) < 5:
            continue
        x = np.log10(valid["params_b_c1"].to_numpy(float))
        y = valid["score"].to_numpy(float)
        fit = stats.linregress(x, y)
        pred = fit.intercept + fit.slope * x
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / total if total > 0 else np.nan
        rows.append({"task": str(task), "n": int(len(valid)),
                     "scale_slope_points_per_log10_params": float(fit.slope),
                     "r2": float(r2),
                     "spearman_rho": float(stats.spearmanr(x, y).statistic),
                     "params_coverage": float(len(valid) / len(group))})
    return pd.DataFrame(rows, columns=columns).sort_values("task").reset_index(drop=True)


def c4_field_profile(epoch: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage and association of required C4 fields."""
    d = epoch.copy()
    n = len(d)
    rows = []
    for col in ["Training compute (FLOP)", "Training dataset size (total)", "Parameters",
                "Open model weights?", "Frontier model", "Publication date",
                "Model accessibility", "Training code accessibility"]:
        if col not in d:
            continue
        s = d[col]
        rows.append({"field": col, "nonmissing": int(s.notna().sum()),
                     "total_rows": n, "coverage": float(s.notna().mean()),
                     "distinct_nonmissing": int(s.dropna().nunique())})
    return pd.DataFrame(rows)


def c3_annual_envelope(timeseries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual envelopes and explicitly diagnose forecast identifiability.

    With only three annual points, the ordinary t prediction interval has one
    residual degree of freedom and expands beyond the physical [0, 100] score
    domain.  We retain that raw interval for auditability, but mark it as
    non-identifiable and leave the bounded interval fields missing so it cannot
    be mistaken for an informative uncertainty band.
    """
    d = timeseries.copy()
    d["Year"] = pd.to_numeric(d.Year, errors="coerce")
    d["Average"] = pd.to_numeric(d.Average, errors="coerce")
    d = d.dropna(subset=["Year", "Average"])
    annual = d.groupby("Year").Average.agg(n="count", mean="mean", p90=lambda x: x.quantile(.9), maximum="max").reset_index()
    fit_data = annual[annual.n >= 10]
    rows = []
    if len(fit_data) >= 3:
        for metric in ["p90", "maximum"]:
            x, y = fit_data.Year.to_numpy(float), fit_data[metric].to_numpy(float)
            fit = stats.linregress(x, y)
            for horizon in [1, 2]:
                year = float(x.max() + horizon)
                pred = fit.intercept + fit.slope * year
                dof = max(1, len(x)-2)
                residual_se = np.sqrt(np.sum((y-(fit.intercept+fit.slope*x))**2)/dof)
                se_pred = residual_se * np.sqrt(1 + 1/len(x) + (year-x.mean())**2/np.sum((x-x.mean())**2))
                crit = stats.t.ppf(.975, dof)
                raw_low = pred - crit * se_pred
                raw_high = pred + crit * se_pred
                identifiable = len(x) >= 5 and dof >= 3
                rows.append({
                    "metric": metric,
                    "forecast_year": int(year),
                    "estimate": pred,
                    "raw_prediction_low_95": raw_low,
                    "raw_prediction_high_95": raw_high,
                    # Kept for schema compatibility; NaN prevents false precision.
                    "prediction_low_95": raw_low if identifiable else np.nan,
                    "prediction_high_95": raw_high if identifiable else np.nan,
                    "physical_low": 0.0,
                    "physical_high": 100.0,
                    "interval_status": "usable_t_interval" if identifiable else "not_identifiable_df1",
                    "t_dof": dof,
                    "slope_points_per_year": fit.slope,
                    "n_annual_points": len(x),
                })
    return annual, pd.DataFrame(rows)
