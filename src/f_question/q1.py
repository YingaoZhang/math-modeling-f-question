"""Reproducible analysis for Problem 1 of the LLM resource allocation task."""

from __future__ import annotations

import argparse
import json
import lzma
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.linalg import helmert
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .diagnostics import (conflict_plot, diagnose_classifier_domain_shift,
                          inspect_a17_a18, run_conflict_diagnostics)
from .extrapolation import (analyze_est_data_generation_mechanism,
                            analyze_estimated_scales,
                            evaluate_exponential_split_sets)
from .bridge import export_q2_bridge_interface
from .quality_extensions import run_quality_extensions
from .reporting import write_supplemental_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "real_attachments"
OUT = PROJECT_ROOT / "outputs" / "q1"
GROUPS = ["edu", "read", "reason", "clean", "struct"]
LOSS_PREFIX = "metric/the_pile_"
GROUP_FIELDS = {
    "edu": [("fineweb_edu", 1)],
    "read": [("modernbert_readability", 1), ("fluency_en", 1)],
    "reason": [("modernbert_reasoning", 1), ("modernbert_professionalism", 1)],
    "clean": [("modernbert_cleanliness", 1), ("ad_en", -1)],
    "struct": [("rps_lines_ending_with_terminal_punctution_mark", 1),
                ("rps_doc_frac_no_alph_words", -1), ("rps_doc_frac_chars_top_2gram", -1),
                ("rps_doc_frac_chars_top_3gram", -1), ("rps_lines_uppercase_letter_fraction", -1),
                ("rps_doc_frac_unique_words", 1), ("rps_lines_numerical_chars_fraction", -1),
                ("rps_doc_unigram_entropy", 1)],
}


def read_xz_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def domain_from_row(row: pd.Series, fallback: str = "unknown") -> str:
    value = row.get("_source_domain")
    if isinstance(value, str) and value:
        return value.lower()
    value = row.get("sub_path")
    if isinstance(value, str) and value:
        return value.split("/")[0].lower()
    return fallback


def scalarize(value, field: str) -> float:
    if value is None:
        return np.nan
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return float(arr[0])
        probs = softmax(arr)
        if field.startswith("modernbert_") and arr.size == 6:
            return float(np.dot(np.arange(6), probs))
        if field in {"fluency_en", "ad_en"} and arr.size == 2:
            return float(probs[1])
        return float(np.nanmean(arr))
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def quality_raw_frame(path: Path, fallback: str = "unknown") -> pd.DataFrame:
    raw = read_xz_jsonl(path)
    records = []
    fields = [
        "fineweb_edu", "modernbert_readability", "fluency_en",
        "modernbert_reasoning", "modernbert_professionalism",
        "modernbert_cleanliness", "ad_en",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
        "rps_doc_frac_unique_words", "rps_lines_numerical_chars_fraction",
        "rps_doc_unigram_entropy", "rps_doc_num_sentences",
        "rps_doc_word_count", "rps_doc_mean_word_length",
        "dsir_books", "dsir_math", "dsir_wiki", "qurater",
    ]
    for _, row in raw.iterrows():
        rec = {f: scalarize(row.get(f), f) for f in fields}
        rec["domain"] = domain_from_row(row, fallback)
        rec["id"] = row.get("id", "")
        records.append(rec)
    return pd.DataFrame(records)


def empirical_cdf(values: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    ref = x.dropna() if reference is None else pd.to_numeric(reference, errors="coerce").dropna()
    if len(ref) == 0:
        return pd.Series(np.nan, index=x.index)
    ref_sorted = np.sort(ref.to_numpy())
    ranks = np.searchsorted(ref_sorted, x.to_numpy(dtype=float), side="right")
    result = (ranks + 0.5) / (len(ref_sorted) + 1.0)
    result[~np.isfinite(x.to_numpy(dtype=float))] = np.nan
    return pd.Series(result, index=x.index).clip(1e-9, 1.0)


def score_quality(raw: pd.DataFrame, references: dict[str, pd.Series] | None = None):
    d = raw.copy()
    group_fields = GROUP_FIELDS
    metric_refs = {}
    for group, entries in group_fields.items():
        cols = []
        for field, direction in entries:
            source = -d[field] if direction < 0 else d[field]
            ref = None if references is None else references.get(field)
            d[field + "__q"] = empirical_cdf(source, ref)
            cols.append(field + "__q")
            if references is None:
                metric_refs[field] = source.dropna().copy()
        d[group] = d[cols].mean(axis=1, skipna=True)
    d["Q_equal"] = d[GROUPS].mean(axis=1, skipna=True).clip(1e-9, 1.0)
    return d, metric_refs


def critic_weights(frame: pd.DataFrame) -> pd.Series:
    x = frame[GROUPS].copy()
    x = (x - x.min()) / (x.max() - x.min()).replace(0, np.nan)
    x = x.fillna(0.5)
    std = x.std(ddof=1).fillna(0.0)
    corr = x.corr().abs().fillna(0.0)
    conflict = 1.0 - (corr.sum(axis=1) - 1.0) / max(1, len(GROUPS) - 1)
    raw = (std * conflict).clip(lower=0)
    if raw.sum() == 0:
        return pd.Series(1 / len(GROUPS), index=GROUPS)
    return raw / raw.sum()


def entropy_weights(frame: pd.DataFrame) -> pd.Series:
    x = frame[GROUPS].copy()
    x = (x - x.min()) / (x.max() - x.min()).replace(0, np.nan)
    x = x.fillna(0.0).clip(lower=0)
    p = x.div(x.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    n = max(2, len(p))
    entropy = -(p.where(p > 0, 1) * np.log(p.where(p > 0, 1))).sum(axis=0) / np.log(n)
    raw = (1 - entropy).clip(lower=0)
    return raw / raw.sum() if raw.sum() else pd.Series(1 / len(GROUPS), index=GROUPS)


def score_with_weights(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = frame[GROUPS]
    numerator = x.mul(weights, axis=1).sum(axis=1, min_count=1)
    denominator = x.notna().mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
    return (numerator / denominator).clip(1e-9, 1.0)


def bootstrap_mean_ci(values: pd.Series, n_boot: int = 2000, seed: int = 20260923):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed + len(x))
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(x, size=len(x), replace=True).mean()
    return np.quantile(means, [0.025, 0.975])


def compositional_replace(x: np.ndarray, delta: float) -> np.ndarray:
    """Multiplicative replacement of true zero parts followed by closure."""
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    x = x / x.sum(axis=1, keepdims=True)
    out = x.copy()
    for i, row in enumerate(out):
        zeros = row <= 0
        count = int(zeros.sum())
        if count == 0:
            continue
        if count * delta >= 1:
            raise ValueError(f"delta={delta} too large for row {i} with {count} zeros")
        positive_sum = row[~zeros].sum()
        if positive_sum <= 0:
            raise ValueError(f"row {i} has no positive composition parts")
        out[i, zeros] = delta
        out[i, ~zeros] *= (1.0 - count * delta) / positive_sum
    return out / out.sum(axis=1, keepdims=True)


def ilr_transform(x: np.ndarray, delta: float) -> np.ndarray:
    closed = compositional_replace(x, delta)
    basis = helmert(closed.shape[1], full=False).T
    return np.log(closed) @ basis


def load_mixture_pair(data_root: Path, split: str):
    base = data_root / "A_data_value" / "regmix_tables"
    mix_path = base / f"{split}_mixture_1m.csv"
    loss_path = base / f"{split}_pile_loss_1m.csv"
    xdf = pd.read_csv(mix_path).set_index("index")
    ydf = pd.read_csv(loss_path).set_index("index")
    common = xdf.index.intersection(ydf.index)
    xdf, ydf = xdf.loc[common], ydf.loc[common]
    cols = [c for c in xdf if c.startswith("train_the_pile_")]
    ycols = [c for c in ydf if c.startswith(LOSS_PREFIX) and c.endswith("_val_loss")]
    return xdf[cols].to_numpy(float), ydf[ycols], cols, ycols


def make_model(poly: bool = False):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("ridge", Ridge()))
    return Pipeline(steps)


def dm_law_predict(X_train, y_train, X_test, alpha, return_params=False):
    """Fit L = c + k exp(t dot p), with L2 shrinkage on t."""
    basis = helmert(X_train.shape[1], full=False).T
    xtr = compositional_replace(X_train, 1e-4) @ basis
    xte = compositional_replace(X_test, 1e-4) @ basis
    y_scale = max(float(np.std(y_train)), 1e-6)

    def residual(theta, x, y):
        c, log_k = theta[:2]
        t = theta[2:]
        pred = c + np.exp(np.clip(log_k + x @ t, -30, 30))
        return np.r_[(pred - y) / y_scale, np.sqrt(alpha) * t]

    init = np.r_[max(0.0, float(np.min(y_train)) * 0.5), np.log(y_scale), np.zeros(xtr.shape[1])]
    lo = np.r_[0.0, -20.0, np.full(xtr.shape[1], -20.0)]
    hi = np.r_[max(20.0, float(np.max(y_train))), 20.0, np.full(xtr.shape[1], 20.0)]
    fit = least_squares(residual, init, args=(xtr, y_train), bounds=(lo, hi), max_nfev=250)
    c, log_k, *theta = fit.x
    t = basis @ np.asarray(theta)
    pred = c + np.exp(np.clip(log_k + xte @ np.asarray(theta), -30, 30))
    if return_params:
        return pred, {"c": c, "k": float(np.exp(log_k)), "t": t}
    return pred


def fit_domain_models(data_root: Path):
    Xtr, Ytr, xcols, ycols = load_mixture_pair(data_root, "train")
    domains = [c.replace("train_the_pile_", "") for c in xcols]
    alpha_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    all_results, preds, residuals, model_rows, model_compare, dm_coefficients, joint_effects = [], {}, {}, [], [], [], []
    alpha_values = [1.0, 10.0, 100.0, 1000.0]
    for ycol in ycols:
        search = GridSearchCV(make_model(), alpha_grid, scoring="neg_mean_squared_error", cv=cv)
        search.fit(ilr_transform(Xtr, 1e-4), Ytr[ycol].to_numpy(float))
        model_rows.append({"loss_domain": ycol, "best_alpha": search.best_params_["ridge__alpha"],
                           "cv_rmse": np.sqrt(-search.best_score_)})
        for split, x_name, y_name in [("test", "test_mixture_1m.csv", "test_pile_loss_1m.csv"),
                                      ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
                                      ("test_1b", "test_mixture_1B.csv", "test_pile_loss_1B.csv")]:
            base = data_root / "A_data_value" / "regmix_tables"
            xx = pd.read_csv(base / x_name).set_index("index")
            yy = pd.read_csv(base / y_name).set_index("index")
            common = xx.index.intersection(yy.index)
            xcols_here = [c for c in xx if c.startswith("train_the_pile_")]
            xt = ilr_transform(xx.loc[common, xcols_here].to_numpy(float), 1e-4)
            yt = yy.loc[common, ycol].to_numpy(float)
            yp = search.predict(xt)
            key = f"{split}::{ycol}"
            preds[key] = (yt, yp)
            residuals[key] = yt - yp
            rho = pd.Series(yt).corr(pd.Series(yp), method="spearman")
            all_results.append({"split": split, "loss_domain": ycol, "n": len(yt),
                                "r2": r2_score(yt, yp),
                                "rmse": mean_squared_error(yt, yp) ** 0.5,
                                "mae": mean_absolute_error(yt, yp), "spearman": rho})

    # Model selection is confined to training CV; the held-out same-scale test
    # is used once for comparison. PolynomialFeatures includes square and
    # pairwise interaction terms, with Ridge controlling the 152-term design.
    base = data_root / "A_data_value" / "regmix_tables"
    Xtest_df = pd.read_csv(base / "test_mixture_1m.csv").set_index("index")
    Ytest = pd.read_csv(base / "test_pile_loss_1m.csv").set_index("index")
    common = Xtest_df.index.intersection(Ytest.index)
    Xtest_raw = Xtest_df.loc[common, [c for c in Xtest_df if c.startswith("train_the_pile_")]].to_numpy(float)
    Xtrain_ilr = ilr_transform(Xtr, 1e-4)
    Xtest_ilr = ilr_transform(Xtest_raw, 1e-4)
    for ycol in ycols:
        y = Ytr[ycol].to_numpy(float)
        yt = Ytest.loc[common, ycol].to_numpy(float)
        quadratic_model = None
        for label, poly in [("linear_ridge", False), ("quadratic_interaction_ridge", True)]:
            grid = {"ridge__alpha": alpha_values}
            cv_search = GridSearchCV(make_model(poly), grid, scoring="neg_mean_squared_error", cv=cv)
            cv_search.fit(Xtrain_ilr, y)
            model = cv_search.best_estimator_
            yp = model.predict(Xtest_ilr)
            if poly:
                quadratic_model = model
            model_compare.append({"loss_domain": ycol, "model": label,
                                  "best_alpha": cv_search.best_params_["ridge__alpha"],
                                  "cv_rmse": np.sqrt(-cv_search.best_score_),
                                  "r2": r2_score(yt, yp),
                                  "rmse": mean_squared_error(yt, yp) ** 0.5,
                                  "spearman": pd.Series(yt).corr(pd.Series(yp), method="spearman")})
        # Non-additivity of two simultaneous 5 pp transfers, evaluated only
        # for feasible independent test recipes and regularized quadratic fits.
        mixture_names = [c.replace("train_the_pile_", "") for c in xcols]
        mean_share = Xtr.mean(axis=0)
        common_domains = np.argsort(mean_share)[-8:]
        for donor in common_domains:
            recipients = [j for j in common_domains if j != donor]
            for ai, first in enumerate(recipients):
                for second in recipients[ai + 1:]:
                    feasible = Xtest_raw[:, donor] >= 0.10
                    if not feasible.any():
                        continue
                    base_recipe = Xtest_raw[feasible].copy()
                    p_ab = base_recipe.copy(); p_ab[:, donor] -= 0.10; p_ab[:, first] += 0.05; p_ab[:, second] += 0.05
                    p_a = base_recipe.copy(); p_a[:, donor] -= 0.05; p_a[:, first] += 0.05
                    p_b = base_recipe.copy(); p_b[:, donor] -= 0.05; p_b[:, second] += 0.05
                    pred_base = quadratic_model.predict(ilr_transform(base_recipe, 1e-4))
                    pred_ab = quadratic_model.predict(ilr_transform(p_ab, 1e-4))
                    pred_a = quadratic_model.predict(ilr_transform(p_a, 1e-4))
                    pred_b = quadratic_model.predict(ilr_transform(p_b, 1e-4))
                    nonadd = (pred_ab - pred_base) - ((pred_a-pred_base) + (pred_b-pred_base))
                    joint_effects.append({"loss_domain": ycol, "donor_domain": mixture_names[donor],
                        "recipient_1": mixture_names[first], "recipient_2": mixture_names[second],
                        "transfer_each": 0.05, "n_test_recipes": len(nonadd),
                        "mean_nonadditive_delta_loss": float(np.mean(nonadd)),
                        "median_nonadditive_delta_loss": float(np.median(nonadd)),
                        "mean_absolute_nonadditive_delta_loss": float(np.mean(np.abs(nonadd))),
                        "sign_consistency": float(max(np.mean(nonadd > 0), np.mean(nonadd < 0))),
                        "interpretation": "quadratic-model non-additivity; conditional association, not causal synergy"})
        best_dm = (None, np.inf)
        for alpha in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            fold_losses = []
            for tr_idx, va_idx in cv.split(Xtr):
                pred = dm_law_predict(Xtr[tr_idx], y[tr_idx], Xtr[va_idx], alpha)
                fold_losses.append(mean_squared_error(y[va_idx], pred))
            if np.mean(fold_losses) < best_dm[1]:
                best_dm = (alpha, np.mean(fold_losses))
        dm_pred, dm_params = dm_law_predict(Xtr, y, Xtest_raw, best_dm[0], return_params=True)
        for d, coefficient in zip(domains, dm_params["t"]):
            dm_coefficients.append({"loss_domain": ycol, "mixture_domain": d,
                                    "t_relative_substitution": coefficient,
                                    "c": dm_params["c"], "k": dm_params["k"],
                                    "sum_t_constraint": "sum(t)=0"})
        model_compare.append({"loss_domain": ycol, "model": "data_mixing_exponential",
                              "best_alpha": best_dm[0], "cv_rmse": np.sqrt(best_dm[1]),
                              "r2": r2_score(yt, dm_pred),
                              "rmse": mean_squared_error(yt, dm_pred) ** 0.5,
                              "spearman": pd.Series(yt).corr(pd.Series(dm_pred), method="spearman")})
    return (pd.DataFrame(model_rows), pd.DataFrame(all_results), pd.DataFrame(model_compare),
            preds, residuals, pd.DataFrame(dm_coefficients), pd.DataFrame(joint_effects))


def make_plots(quality: pd.DataFrame, predictions: dict, residuals: dict):
    OUT.mkdir(exist_ok=True)
    domains = sorted(quality["domain"].dropna().unique())
    samples = [quality.loc[quality.domain == d, "Q_equal"].dropna() for d in domains]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(samples, tick_labels=domains, showfliers=False)
    ax.set_ylabel("Q (global empirical-CDF scale)")
    ax.set_title("Document quality score by quality-signal domain")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(OUT / "q_domain_boxplot.png", dpi=180); plt.close(fig)

    focus = [d for d in ["github", "arxiv"] if d in domains]
    if len(focus) == 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [quality.loc[quality.domain == d, "Q_equal"].dropna().to_numpy() for d in focus]
        ax.boxplot(vals, tick_labels=focus, showfliers=False, widths=0.55)
        ax.set_ylabel("Document quality Q")
        ax.set_title("Quality-score distributions: GitHub and arXiv")
        fig.tight_layout(); fig.savefig(OUT / "q_github_arxiv_comparison.png", dpi=220); plt.close(fig)

    keys = [k for k in predictions if k.startswith("test::")]
    ncols = 4; nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7*nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        y, p = predictions[key]
        ax.scatter(y, p, s=12, alpha=.6)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1)
        ax.set_title(key.split("/")[-1].replace("_val_loss", ""), fontsize=8)
        ax.set_xlabel("Observed Loss"); ax.set_ylabel("Predicted Loss")
    for ax in axes.flat[len(keys):]: ax.remove()
    fig.tight_layout(); fig.savefig(OUT / "loss_scatter_by_domain.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7*nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        y, p = predictions[key]; r = residuals[key]
        ax.scatter(p, r, s=12, alpha=.6)
        ax.axhline(0, color="crimson", linewidth=1)
        ax.set_title(key.split("/")[-1].replace("_val_loss", ""), fontsize=8)
        ax.set_xlabel("Predicted Loss"); ax.set_ylabel("Residual")
    for ax in axes.flat[len(keys):]: ax.remove()
    fig.tight_layout(); fig.savefig(OUT / "loss_residuals_by_domain.png", dpi=180); plt.close(fig)

    # Aggregate only after within-domain standardization; no raw domain losses
    # are averaged, so the plot cannot be dominated by a high-loss domain.
    z_obs, z_pred = [], []
    for key in keys:
        y, p = predictions[key]
        sy, sp = np.std(y), np.std(p)
        if sy > 0 and sp > 0:
            z_obs.extend((y - np.mean(y)) / sy)
            z_pred.extend((p - np.mean(p)) / sp)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(z_obs, z_pred, s=9, alpha=0.18, color="#2878A5", edgecolors="none")
    lim = max(np.max(np.abs(z_obs)), np.max(np.abs(z_pred)))
    ax.plot([-lim, lim], [-lim, lim], color="#B64A4A", linewidth=1.4)
    ax.set(xlabel="Observed Loss (within-domain z-score)", ylabel="Predicted Loss (within-domain z-score)",
           title="Pooled same-scale test after domain standardization")
    fig.tight_layout(); fig.savefig(OUT / "loss_scatter_standardized_aggregate.png", dpi=220); plt.close(fig)


def plot_conflict_sensitivity():
    data = pd.read_csv(OUT / "conflict_threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for domain, group in data.groupby("domain"):
        group = group.sort_values("upper_quantile")
        ax.plot(group.upper_quantile * 100, group.conflict_rate * 100, marker="o", linewidth=1.2, label=domain)
    ax.set(xlabel="Upper/lower within-domain percentile threshold",
           ylabel="Documents flagged as conflicting (%)",
           title="Conflict-rate sensitivity by quality domain")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "conflict_threshold_curve.png", dpi=220); plt.close(fig)


def write_report(q_summary, critic, entropy, metrics, model_compare):
    same = metrics.loc[metrics.split == "test"]
    cross60 = metrics.loc[metrics.split == "test_60m"]
    cross1b = metrics.loc[metrics.split == "test_1b"]
    model_summary = model_compare.groupby("model").agg(
        macro_R2=("r2", "mean"), median_domain_Spearman=("spearman", "median"),
        domains_R2_positive=("r2", lambda x: int((x > 0).sum()))).reset_index()
    winners = model_compare.loc[model_compare.groupby("loss_domain").cv_rmse.idxmin(), ["loss_domain", "model", "best_alpha", "cv_rmse", "r2"]]
    winner_counts = winners.model.value_counts()
    weight_table = pd.DataFrame({"group": GROUPS, "equal": [0.2] * len(GROUPS),
                                 "CRITIC": critic.reindex(GROUPS).to_numpy(),
                                 "entropy": entropy.reindex(GROUPS).to_numpy()})
    q_table = q_summary[["domain", "n", "Q_equal_mean", "Q_CI_low", "Q_CI_high", "Q_length_weighted"]]
    def markdown_table(frame):
        def cell(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return str(value).replace("|", "\\|")
        lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
                 "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(lines)
    extension_q = pd.read_csv(OUT / "q_A1_A2_A3_domain_comparison.csv") if (OUT / "q_A1_A2_A3_domain_comparison.csv").exists() else pd.DataFrame()
    extension_domains = pd.read_csv(OUT / "q_domain_summary_A1_A2_A3_all.csv") if (OUT / "q_domain_summary_A1_A2_A3_all.csv").exists() else pd.DataFrame()
    rank_stability = pd.read_csv(OUT / "q_domain_rank_stability.csv") if (OUT / "q_domain_rank_stability.csv").exists() else pd.DataFrame()
    conflict_by_pop = pd.read_csv(OUT / "conflict_threshold_by_population.csv")
    conflict_pairs = pd.read_csv(OUT / "conflict_pair_top10.csv")
    penalty = pd.read_csv(OUT / "conflict_penalty_sensitivity.csv")
    a18_examples = pd.read_csv(OUT / "A18_qualitative_text_examples.csv")
    jt_path = OUT.parent / "validation" / "A17_A18_quintile_jt_validation.csv"
    jt_summary_path = OUT.parent / "validation" / "A17_A18_quintile_summary.csv"
    if jt_path.exists():
        jt = pd.read_csv(jt_path).iloc[0]
        jt_text = (f"补充的 A17/A18 五分位有序验证按 A17 prior share 将样本分为 {int(jt['quintiles'])} 组，"
                   f"以 A18 脱敏文本字符数作为文本量，Jonckheere–Terpstra 统计量 J={jt['J_stat']:.0f}，"
                   f"2000 次置换 p={jt['permutation_p']:.4f}。该检验只说明文本量在先验分位间的探索性有序关系，"
                   "A18 没有质量标签，因此不把它解释为质量因果验证或数值 Q 的外部校准。")
    else:
        jt_text = "A17/A18 五分位有序验证结果将在验证输出生成后写入。"
    joint_effects = pd.read_csv(OUT / "mixture_joint_transfer_effects.csv") if (OUT / "mixture_joint_transfer_effects.csv").exists() else pd.DataFrame()
    extrap = pd.read_csv(OUT / "estimated_loss_ranking_reversal_summary.csv") if (OUT / "estimated_loss_ranking_reversal_summary.csv").exists() else pd.DataFrame()
    scaling = pd.read_csv(OUT / "scaling_est_reverse_engineering.csv") if (OUT / "scaling_est_reverse_engineering.csv").exists() else pd.DataFrame()
    appendix_eval = pd.read_csv(OUT / "appendix_split_exponential_metrics_summary.csv") if (OUT / "appendix_split_exponential_metrics_summary.csv").exists() else pd.DataFrame()
    shift = pd.read_csv(OUT / "conflict_classifier_domain_shift.csv") if (OUT / "conflict_classifier_domain_shift.csv").exists() else pd.DataFrame()
    if len(joint_effects):
        joint_summary = joint_effects.groupby("loss_domain", as_index=False).agg(
            evaluated_combinations=("mean_absolute_nonadditive_delta_loss", "size"),
            median_abs_nonadditive_delta=("mean_absolute_nonadditive_delta_loss", "median"),
            median_sign_consistency=("sign_consistency", "median"))
        top_rows = joint_effects.groupby("loss_domain").mean_absolute_nonadditive_delta_loss.idxmax()
        joint_top_within_domain = joint_effects.loc[top_rows].sort_values("loss_domain")
    else:
        joint_summary = joint_top_within_domain = joint_effects
    body = f"""# 数学建模 F 题第一问：完整计算报告

## 研究目标与结论

第一问分为语料质量评价、质量冲突识别、配比到验证损失建模三部分。主分析遵守三个口径：质量分 Q 属于 (0,1]；配比先闭合到单纯形再做 ILR；13 个验证域 Loss 各自建模，不对原始 Loss 求算术平均。

1. A1 质量信号样本共 {int(q_summary.n.sum()):,} 条，覆盖 {len(q_summary)} 个域。等权组分 Q 保持有界，域级 bootstrap 区间见下表。
2. 13 个 The Pile 验证域分别拟合。CV 选型宏平均 R²={model_compare.r2.mean():.4f}，纯指数模型宏平均 R²={model_compare.loc[model_compare.model == 'data_mixing_exponential', 'r2'].mean():.4f}；两者均先逐域计算，再对域级 R² 求平均。
3. 1M 训练到 60M 测试的宏平均 R²={cross60.r2.mean():.3f}、Spearman 中位数={cross60.spearman.median():.4f}；到 1B 的宏平均 R²={cross1b.r2.mean():.3f}、Spearman 中位数={cross1b.spearman.median():.4f}。绝对 Loss 不具跨尺度迁移性，配比排序信号相对稳定。
4. 同尺度检验是预先指定的模型形式比较集；模型形式与惩罚参数均通过训练集 CV 逐域选择。
5. 逐域 CV 选型中，{winner_counts.index[0]} 在 {int(winner_counts.iloc[0])}/13 个域获选；所选模型的独立同尺度测试宏平均 R²={model_compare.loc[model_compare.groupby('loss_domain').cv_rmse.idxmin(), 'r2'].mean():.4f}，13 域中正 R² 数={int((model_compare.loc[model_compare.groupby('loss_domain').cv_rmse.idxmin(), 'r2'] > 0).sum())}。详细逐域选择表见 `selected_model_by_domain.csv`。

## 质量分定义

每个原始指标依题意统一方向，经 A1 参考经验 CDF 映射到 (0,1]。多类别分类器先 softmax 后取期望，二分类取正类概率，单元素 `fineweb_edu` 列表直接取值。指标按语义归为 edu、read、reason、clean、struct 五组，组内缺失值可用指标均值聚合，组间按权重求和：

$$Q_i=\\sum_g \\omega_g Q_{{ig}},\\quad Q_i\\in(0,1],\\quad \\sum_g\\omega_g=1.$$

等权用于主口径；CRITIC 和熵权仅为无监督敏感性对照。监督式 Meta-rater 权重作为后续校准接口保留，当前以两阶段工程逼近处理配对不足。A2/A3 全量与 A1 对应域部分重合，因此同时报告全量域级 Q 及按 id 去重后的非重叠结果；两者使用冻结的 A1 指标经验 CDF 标尺。

### 域级质量分和区间

{markdown_table(q_table)}

### 三套权重

{markdown_table(weight_table)}

长度加权仅作为文档贡献口径敏感性分析，见 `q_length_weighting_comparison.csv`，不改变文档等权主分析。

### 全量扩展集域级 Q 与排序稳定性

下表报告 A2/arxiv、A3/github 全量记录，以及剔除与 A1 重合记录后的域级 Q。各分数仍在 (0,1]，CI 为文档 bootstrap 区间。

{markdown_table(extension_domains.loc[extension_domains.weighting == "equal", ["dataset", "domain", "n", "Q_mean", "Q_CI_low", "Q_CI_high"]])}

与 A1 对应域的量表差值比较：

{markdown_table(extension_q)}

权重和长度口径变化下的域排名稳定性：

{markdown_table(rank_stability)}

A1 的 7 个域中，CRITIC 排名与等权完全一致；熵权和长度加权的 Spearman 均为 0.964，最大只移动 1 位。A2/A3 每个扩展文件仅覆盖一个域，因此其域间排序相关不适用；表中空 Spearman 表示样本域数为 1，不是缺失计算。

## 冲突识别

先在每个域内对每个质量组按文档计算经验分位，再定义冲突：至少一个组达到上阈值，且另一个组低于下阈值。报告阈值 (P70,P30)、(P75,P25)、(P80,P20)、(P90,P10) 的逐域冲突率；完整曲线见 `conflict_threshold_sensitivity.csv`。域内标准化避免把域先验错判为文档内冲突。主口径切换为 P90/P10，因为它只保留极端尾部。

综合评价使用工程惩罚分 $Q_i^*=\\operatorname{{clip}}(Q_i-\\rho C_i,\\epsilon,1)$，其中 $C_i$ 为五组域内秩相对中位秩的平均绝对偏差归一化值，主值 rho=0.25，并报告 rho=0/0.1/0.5/1.0 的敏感性。现有资料尚未形成文档 Q 与配方 Loss 的逐条配对，因此该项定位为稳健界定和跨问接口校准，而非监督增益宣称。

扩展集 P75/P25 冲突率：

{markdown_table(conflict_by_pop.query("upper_quantile == 0.75").groupby("dataset", as_index=False).agg(records=("n", "sum"), conflicts=("conflict_n", "sum"), conflict_rate=("conflict_rate", "mean")))}

主要组对 (P75/P25，按各数据集前列组合)：

{markdown_table(conflict_pairs.groupby("dataset", as_index=False, group_keys=False).head(3))}

冲突惩罚的 λ 敏感性结果：

{markdown_table(penalty)}

分类器域偏移诊断将 ModernBERT、FineWeb-Edu 与 Qurater 聚合为 $G_{{model}}$，将结构统计聚合为 $G_{{struct}}$，并用 $G_{{noise}}=1-ad_en$ 表示清洁方向。以下表格给出 arxiv 与 stackexchange 的离差和共现率；这是跨域失配的风险证据，缺少人工广告真值时不能直接命名为“误判率”。因此支持对广告信号做门控或降权，不支持无条件删除文档。

{markdown_table(shift.loc[shift.domain.isin(['arxiv','stackexchange']), ['domain','G_model_mean','G_noise_mean','G_model_minus_global','G_noise_minus_global','model_high_ad_high_rate']] if len(shift) else pd.DataFrame())}

置换检验改变了结论。将五组在域内独立重排 1000 次后，独立基准约为 55.7%；实测域等权冲突率为 A1 42.6%、A2 53.7%、A3 47.9%，均位于零分布的显著下侧。因此各质量维度总体同向，冲突只属少数情形，不应再写成“冲突普遍存在”。P90/P10 下冲突率约为 9.1% 的量级，极端尾部几乎不冲突。完整零分布见 `conflict_permutation_null_draws.csv.gz`，检验表见 `conflict_permutation_test.csv`；22 指标直接聚合的敏感性对照见 `conflict_all22_sensitivity.csv`。

## 配比到 Loss

给定配方原始比例 p，先按行闭合 `p_j <- p_j / sum(p)`，再将零值乘性替换为 δ 并重新闭合，使用 Helmert 正交基得到 16 维 ILR 坐标 z。线性 Ridge 基线为：

$$L_d = \\beta_{{0d}} + z^T\\beta_d + \\epsilon_d,\\qquad d=1,\\ldots,13.$$

每个域单独 5 折 CV 选择 Ridge 惩罚强度。17 个组成只有 16 个自由度；ILR 系数是对数对比，任一域比例上升必从其他域替代而来，因此系数表达相对替代效应，不是绝对边际效应。4 个无对应 Loss 的组成仍会通过闭合与基准对比影响系数。

δ 敏感性、单纯形行和误差、逐域测试指标、模型形式比较和系数语义表均单独导出。交互模型用 ILR 的平方项和两两交互，共 152 个二阶设计项，并用 CV Ridge 收缩。独立检验配方上另计算固定份额联合替代：仅取训练集中平均占比最高的 8 个组成域作为有数据支持的分析范围，从一个来源域各转移 5 个百分点到两个目标域，对照两个单独转移效应之和；非加性差值描述二次模型中的组合关联，不是全 17 域无差别的因果结论。

### 独立同尺度测试上的模型比较

{markdown_table(model_summary)}

独立检验集上的联合替代非加性效应按 Loss 域分别汇总；13 个域的绝对差值不跨域排序：

{markdown_table(joint_summary)}

各 Loss 域内平均绝对非加性差值最大的组合（仅在域内定位）：

{markdown_table(joint_top_within_domain)}

### 跨尺度检查

绝对误差与排序相关分别报告，不将不同尺度的 Loss 做混合均值。跨尺度检验用于界定适用范围，而不是模型升级的调参集。

对 A12–A15 的共用 63 个配方，逐域 1M 对估算 10B/70B 的 Spearman 中位数分别为 {pd.read_csv(OUT / "estimated_loss_1m_cross_scale_by_domain.csv").query("scale == '10b'").spearman_1m_vs_estimated.median():.3f} / {pd.read_csv(OUT / "estimated_loss_1m_cross_scale_by_domain.csv").query("scale == '70b'").spearman_1m_vs_estimated.median():.3f}；先逐域 z 标准化再等权合成后的排序相关为 {extrap.loc[extrap.comparison.str.contains("1M vs 10b"), "spearman"].iloc[0]:.3f} / {extrap.loc[extrap.comparison.str.contains("1M vs 70b"), "spearman"].iloc[0]:.3f}。该证据提示配方相对排序具有一定保持性，但估算数据不提供跨尺度绝对 Loss 校准。

### 按附录编号的固定指数模型评估

为直接对应附录 A，以下补充实验固定使用 Data Mixing Laws 指数式 $L_d=c_d+k_d\exp(t_d^T p)$：A4–A5 的 1M 配比与 Loss 成对用于训练，域内正则强度沿用 A4–A5 内部五折 CV 结果；A6–A11 分别评估 1M、60M、1B，A12–A15 分别评估估算 10B、70B。每个尺度独立列出逐域 $R^2$ 与 MAE，再报告 13 域指标的算术均值（先逐域计算指标，不合并原始 Loss）。误差放大倍数定义为同域 MAE 相对 A6–A7 同尺度检验 MAE 的比值，再取域中位数。

{markdown_table(appendix_eval)}

`appendix_split_exponential_metrics_by_domain.csv` 给出 13 域完整结果，`appendix_split_exponential_predictions.csv.gz` 保存逐配方预测，图见 `fig_appendix_split_exponential_eval.png`。A6–A11 是未出现在 A4–A5 的新配方，可用于检验；A12–A15 的 63 个配方则与训练表配方逐行重复，因此这些数字只衡量固定配方的估算跨尺度偏差，不能作为独立配方外推精度或实验验证。A8–A11 也是不同模型规模的实测 Loss，明显的绝对误差放大说明 1M Loss 标度不可直接迁移；排序相关与绝对误差应分开解释。

### A12–A15 标度生成机制反演

对 63 个共同配方逐域计算 $\\gamma=-\\log(L_N/L_{{1M}})/\\log(N/1M)$，并对三尺度的 $\\log L$ 做乘性标度拟合。估算表中逐域结果、相关置信区间和真实 1M→60M 对照见 `scaling_est_reverse_engineering.csv` 与 `scaling_est_gamma_by_recipe.csv`，图见 `fig_est_gamma_coupling.png`。估算数据的 gamma(10B) 与 gamma(70B) 高度一致，但与初始 Loss 的相关性按域差异明显；汇总相关是域-配方池化的描述性统计，不能视为独立样本。真实 1M→60M 对照未复现统一的高正相关，因此现有数据不支持“必然存在 r>0.90 的人工耦合”这一强断言；可以确认的是估算三点几乎严格落在乘性曲线上（估算域内中位 log-fit R²={scaling.loc[scaling.dataset == 'estimated', 'median_scaling_fit_r2'].median() if len(scaling) else float('nan'):.4f}），应将其作为估算机制特征而非实验规律。

## 数据边界与可识别性

- A12–A15 是估算数据；10B 与 70B 使用同一组配方 id，不作为独立实验结果。
- A12–A15 与 A4–A5 的 63 个配方配置逐行相同；固定指数式在其上的评估是估算 Loss 的尺度诊断，不是新配方外推检验。
- A2/A3 含有 A1 对应域记录，报告将全量结果用于题目要求的扩展对照，并另列去重非重叠估计；A2/A3 是扩展抽样而非独立外部验证。
- A17 域摘要用于域先验。A18 原始文本样例覆盖 {len(a18_examples)} 个域，以下仅作定性构念核验；A18 不含 22 项质量信号，不能验证数值 Q，也不能作为冲突标签。

{jt_text}

{markdown_table(a18_examples[["domain", "sample_text_excerpt"]])}
- 附件 A Loss 是 The Pile 验证集交叉熵；附件 B Pythia `val_loss` 是另一验证集的口径。没有共同样本或同模型、同检查点的配对 Loss，仿射/单调桥接不能从当前附件识别。附件 A Q 与 B 的 `Q_score` 也没有同文档配对，故不拟合重标定关系。本报告结果不作为第二问接口的已校准数值。
- A12–A15 的逐域排序与域内标准化等权汇总见 `estimated_loss_ranking_reversal_summary.csv`；任何跨域合成均先在域内标准化，原始 13 域算术均值不作排序结论或模型因变量。

## 输出文件

逐域完整数据表、敏感性 CSV、模型指标、残差与预测散点图、域 Q 分布图，以及按附录编号补充的指数模型逐域结果和误差放大图均在 `outputs/q1/`。图中汇总散点只对每个域分别标准化后叠合，未平均原始 Loss。
"""
    (OUT / "第一问_完整建模报告.md").write_text(body, encoding="utf-8")


def run(data_root: Path):
    OUT.mkdir(exist_ok=True)
    a_dir = data_root / "A_data_value"
    a1_path = a_dir / "slimpajama_quality_signal_sample.jsonl.xz"
    a1_raw = quality_raw_frame(a1_path)
    quality, refs = score_quality(a1_raw)
    critic = critic_weights(quality)
    quality["Q_critic"] = score_with_weights(quality, critic)

    rng = np.random.default_rng(20260923)
    q_rows = []
    for d, g in quality.groupby("domain"):
        low, high = bootstrap_mean_ci(g["Q_equal"])
        q_rows.append({"domain": d, "n": len(g), "Q_equal_mean": g.Q_equal.mean(),
                       "Q_CI_low": low, "Q_CI_high": high,
                       "Q_critic_mean": g.Q_critic.mean(),
                       "mean_length": g.rps_doc_word_count.mean()})
    q_summary = pd.DataFrame(q_rows).sort_values("Q_equal_mean", ascending=False)
    q_summary.to_csv(OUT / "q_domain_summary.csv", index=False)
    entropy = entropy_weights(quality)
    pd.DataFrame({"group": GROUPS, "equal_weight": [0.2]*len(GROUPS),
                  "critic_weight": critic.reindex(GROUPS).to_numpy(),
                  "entropy_weight": entropy.reindex(GROUPS).to_numpy(),
                  "supervised_meta_rater_weight": [np.nan] * len(GROUPS)}).to_csv(
                      OUT / "quality_weight_comparison.csv", index=False)
    pd.DataFrame([
        {"analysis": "supervised_meta_rater_weights", "status": "not_identifiable_from_available_pairing",
         "reason": "A16 maps only 6/17 mixture domains; no document-level A quality score is paired with the mixture Loss experiments. Fitting weights would confound domain effects and quality."},
        {"analysis": "A_ThePile_to_B_Pythia_val_loss_bridge", "status": "not_identifiable_from_available_pairing",
         "reason": "The validation corpora differ and no common paired observations are supplied; no affine or monotone bridge is fit."},
        {"analysis": "A_Q_to_B_Q_score_rescaling", "status": "not_identifiable_from_available_pairing",
         "reason": "No same-document paired A quality scores and B Q_score observations are supplied."},
    ]).to_csv(OUT / "interface_identifiability.csv", index=False)
    quality.to_csv(OUT / "quality_scores_A1.csv.gz", index=False, compression="gzip")

    # A2/A3 contain the A1 records; report A1 versus non-overlapping records only.
    represent = []
    for filename, domain in [
        ("slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
        ("slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
    ]:
        path = a_dir / filename
        ext_raw = quality_raw_frame(path, fallback=domain)
        scored, _ = score_quality(ext_raw, references=refs)
        seen = set(a1_raw.loc[a1_raw.domain == domain, "id"].astype(str))
        ext_scored = scored.loc[~scored.id.astype(str).isin(seen)]
        a1_scores = quality.loc[quality.domain == domain, "Q_equal"]
        new_scores = ext_scored["Q_equal"]
        represent.append({"domain": domain, "A1_n": len(a1_scores), "extension_nonoverlap_n": len(new_scores),
                          "A1_mean_Q": a1_scores.mean(), "extension_nonoverlap_mean_Q": new_scores.mean(),
                          "mean_difference_extension_minus_A1": new_scores.mean()-a1_scores.mean()})
    pd.DataFrame(represent).to_csv(OUT / "sampling_representativeness.csv", index=False)

    # Main paper口径启用正的冲突惩罚：22项指标先完成组内ECDF，再以
    # 域内秩离散度形成 C，Q*=clip(Q-rho C)。rho=0.25 为预注册主值，
    # rho=0/0.1/0.5/1.0 仍保留在敏感性表中。
    run_conflict_diagnostics(a_dir, quality, a1_raw, refs, quality_raw_frame, score_quality, OUT,
                             lambda_penalty=0.25)
    pd.read_csv(OUT / "conflict_threshold_by_population.csv").query("dataset == 'A1'").drop(
        columns=["dataset", "conflict_n"]).to_csv(OUT / "conflict_threshold_sensitivity.csv", index=False)
    conflict_plot(OUT / "conflict_threshold_by_population.csv", OUT / "conflict_threshold_curve.png")

    # Long-document sensitivity: document weights are proportional to word count,
    # while each underlying indicator retains the same global CDF scale.
    length_rows = []
    for d, g in quality.groupby("domain"):
        w = pd.to_numeric(g.rps_doc_word_count, errors="coerce").clip(lower=0).fillna(0)
        length_rows.append({"domain": d, "n": len(g), "Q_document_equal": g.Q_equal.mean(),
                            "Q_length_weighted": np.average(g.Q_equal, weights=w) if w.sum() else np.nan,
                            "difference_length_minus_equal": (np.average(g.Q_equal, weights=w) - g.Q_equal.mean()) if w.sum() else np.nan})
    pd.DataFrame(length_rows).to_csv(OUT / "q_length_weighting_comparison.csv", index=False)

    # Explicit coefficient semantics: every ILR coordinate is a log contrast,
    # and all coefficient effects are relative replacements within the simplex.
    mix_cols = load_mixture_pair(data_root, "train")[2]
    domains = [c.replace("train_the_pile_", "") for c in mix_cols]
    observed_domains = {c.replace(LOSS_PREFIX, "").replace("_val_loss", "")
                        for c in load_mixture_pair(data_root, "train")[3]}
    pd.DataFrame({"composition_domain": domains,
                  "loss_observed": [d in observed_domains for d in domains],
                  "coefficient_semantics": ["relative replacement; effect depends on balance contrast" for _ in domains]}).to_csv(
                      OUT / "ilr_domain_semantics.csv", index=False)

    est_domain, est_summary, est_1m_comparison, _ = analyze_estimated_scales(data_root)
    est_domain.to_csv(OUT / "estimated_loss_10b_vs_70b_by_domain.csv", index=False)
    est_summary.to_csv(OUT / "estimated_loss_ranking_reversal_summary.csv", index=False)
    est_1m_comparison.to_csv(OUT / "estimated_loss_1m_cross_scale_by_domain.csv", index=False)
    pd.DataFrame([{"rows_10b": 63, "rows_70b": 63, "common_recipe_ids": 63,
                   "loss_domains_each_scale": len(est_domain), "independent_extrapolations": False,
                   "interpretation": "A12-A15 are estimated; same 63 recipes; full per-domain rank diagnostics are not independent experiments"}]).to_csv(
                       OUT / "estimated_data_boundary.csv", index=False)

    inspect_a17_a18(a_dir, OUT)
    diagnose_classifier_domain_shift(quality, OUT)
    analyze_est_data_generation_mechanism(data_root, OUT)

    # δ sensitivity diagnostics for the training mixture.
    Xtr, _, _, _ = load_mixture_pair(data_root, "train")
    delta_rows=[]
    for delta in (1e-2,1e-3,1e-4):
        z=ilr_transform(Xtr,delta)
        delta_rows.append({"delta":delta,"mean_abs_ILR":np.mean(np.abs(z)),
                           "max_row_sum_error_before_closure":float(np.max(np.abs(np.asarray(Xtr).sum(axis=1)-1)))})
    pd.DataFrame(delta_rows).to_csv(OUT / "ilr_delta_sensitivity.csv",index=False)

    extension_summary = run_quality_extensions(a_dir, quality, a1_raw, refs, GROUP_FIELDS,
                                               critic, entropy, quality_raw_frame, OUT)
    cv_table, metrics, model_compare, preds, residuals, dm_coefficients, joint_effects = fit_domain_models(data_root)
    cv_table.to_csv(OUT / "ridge_cv_by_domain.csv", index=False)
    metrics.to_csv(OUT / "loss_metrics_by_domain_and_scale.csv", index=False)
    model_compare.to_csv(OUT / "model_form_comparison.csv", index=False)
    dm_coefficients.to_csv(OUT / "data_mixing_law_coefficients.csv", index=False)
    joint_effects.to_csv(OUT / "mixture_joint_transfer_effects.csv", index=False)
    export_q2_bridge_interface(OUT, data_root, quality, dm_coefficients,
                               lambda_penalty=0.25, joint_effects=joint_effects)
    winners = model_compare.loc[model_compare.groupby("loss_domain").cv_rmse.idxmin()].copy()
    winners.to_csv(OUT / "selected_model_by_domain.csv", index=False)
    exponential_alphas = (model_compare.loc[model_compare.model == "data_mixing_exponential"]
                          .set_index("loss_domain").best_alpha.astype(float).to_dict())
    exponential_alphas = {key.replace(LOSS_PREFIX, "").replace("_val_loss", ""): value
                          for key, value in exponential_alphas.items()}
    evaluate_exponential_split_sets(data_root, OUT, exponential_alphas, dm_law_predict)
    make_plots(quality, preds, residuals)
    q_summary["Q_length_weighted"] = pd.read_csv(OUT / "q_length_weighting_comparison.csv").set_index("domain").loc[q_summary.domain, "Q_length_weighted"].to_numpy()
    write_report(q_summary, critic, entropy, metrics, model_compare)
    write_supplemental_report(OUT)

    macro = metrics.groupby("split")["r2"].agg(["mean","median","min","max"]).reset_index()
    macro.to_csv(OUT / "loss_macro_metrics.csv", index=False)
    print(f"A1 quality records: {len(quality)}; domains: {quality.domain.nunique()}")
    print("CRITIC group weights:", critic.round(4).to_dict())
    print("Domain quality summary:")
    print(q_summary.to_string(index=False, float_format=lambda x:f"{x:.4f}"))
    print("Loss metrics macro R2 / median per-domain Spearman:")
    print(metrics.groupby("split").agg(macro_R2=("r2","mean"),median_domain_Spearman=("spearman","median")).to_string())
    print("Model form comparison summary:")
    print(model_compare.groupby("model").agg(macro_R2=("r2","mean"),median_domain_Spearman=("spearman","median")).to_string())
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the paper report from existing q1_outputs CSV files")
    args = parser.parse_args()
    if args.report_only:
        q = pd.read_csv(OUT / "q_domain_summary.csv")
        length = pd.read_csv(OUT / "q_length_weighting_comparison.csv")
        q = q.merge(length[["domain", "Q_length_weighted"]], on="domain", how="left")
        w = pd.read_csv(OUT / "quality_weight_comparison.csv").set_index("group")
        write_report(q, w["critic_weight"], w["entropy_weight"],
                     pd.read_csv(OUT / "loss_metrics_by_domain_and_scale.csv"),
                     pd.read_csv(OUT / "model_form_comparison.csv"))
        write_supplemental_report(OUT)
    else:
        run(args.data)
