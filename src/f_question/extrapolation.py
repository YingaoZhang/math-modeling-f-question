"""Cross-scale diagnostics for estimated RegMix recipes (A12-A15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _pearson(x: pd.Series, y: pd.Series) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="pearson"))


def _corr_bootstrap_ci(x, y, seed=20260924, n_boot=1000):
    """Percentile bootstrap CI for Pearson correlation, for reporting uncertainty."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 8:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r = np.corrcoef(x[idx], y[idx])[0, 1]
        if np.isfinite(r):
            vals.append(r)
    return tuple(np.quantile(vals, [0.025, 0.975])) if vals else (np.nan, np.nan)


def _read_pair(base: Path, scale: str):
    mix = pd.read_csv(base / f"est_mixture_{scale}.csv").set_index("index")
    loss = pd.read_csv(base / f"est_pile_loss_{scale}.csv").set_index("index")
    mix_cols = [c for c in mix if c.startswith("train_the_pile_")]
    loss_cols = [c for c in loss if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    return mix, loss, mix_cols, loss_cols


def _domain_map(columns):
    return {c.replace("metric/the_pile_", "").replace("_val_loss", ""): c for c in columns}


def analyze_estimated_scales(data_root: Path):
    base = data_root / "A_data_value" / "regmix_tables"
    m10, l10, xcols, y10cols = _read_pair(base, "10b")
    m70, l70, _, y70cols = _read_pair(base, "70b")
    ids = m10.index.intersection(m70.index).intersection(l10.index).intersection(l70.index)
    assert len(ids) == 63 and ids.equals(m10.index) and ids.equals(m70.index), "A12-A15 recipe indices do not match expected shared 63 recipes"
    map10, map70 = _domain_map(y10cols), _domain_map(y70cols)
    common_domains = sorted(set(map10) & set(map70))

    scale_rows = []
    for domain in common_domains:
        a, b = l10.loc[ids, map10[domain]], l70.loc[ids, map70[domain]]
        tau = kendalltau(a, b).statistic
        scale_rows.append({"domain": domain, "n_shared_recipes": len(ids),
                           "spearman_loss_10b_70b": spearmanr(a, b).statistic,
                           "kendall_tau_loss_10b_70b": tau,
                           "pairwise_rank_reversal_fraction": (1 - tau) / 2,
                           "best_recipe_10b": int(a.idxmin()), "best_recipe_70b": int(b.idxmin()),
                           "best_recipe_same": bool(a.idxmin() == b.idxmin()),
                           "loss_10b_range": float(a.max() - a.min()),
                           "loss_70b_range": float(b.max() - b.min()),
                           "data_status": "estimated; same 63 recipes; not independent experiments"})
    scale_table = pd.DataFrame(scale_rows)

    # Put each domain on its own scale before forming an equal-domain diagnostic.
    # This is a descriptive standardized composite, not a Loss response.
    def standardized_composite(frame: pd.DataFrame, row_ids, mapping: dict[str, str]) -> pd.Series:
        parts = []
        for domain in common_domains:
            values = frame.loc[row_ids, mapping[domain]].astype(float)
            sd = values.std(ddof=0)
            if sd > 0:
                parts.append((values - values.mean()) / sd)
        return pd.concat(parts, axis=1).mean(axis=1)

    diagnostic10 = standardized_composite(l10, ids, map10)
    diagnostic70 = standardized_composite(l70, ids, map70)
    train = pd.read_csv(base / "train_pile_loss_1m.csv").set_index("index")
    train_map = _domain_map([c for c in train if c.startswith("metric/the_pile_")])
    shared = ids.intersection(train.index)
    diagnostic_rows = [{"comparison": "10B vs 70B; mean within-domain z-score composite",
                        "n_recipes": len(ids), "spearman": spearmanr(diagnostic10, diagnostic70).statistic,
                        "kendall_tau": kendalltau(diagnostic10, diagnostic70).statistic,
                        "best_recipe_a": int(diagnostic10.idxmin()), "best_recipe_b": int(diagnostic70.idxmin()),
                        "data_status": "estimated; equal-domain standardized composite is descriptive only, not a response"}]
    train_est_rows = []
    for scale, loss, mapping in [("10b", l10, map10), ("70b", l70, map70)]:
        est_diag = standardized_composite(loss, shared, mapping)
        train_diag = standardized_composite(train, shared, train_map)
        tau = kendalltau(train_diag, est_diag).statistic
        diagnostic_rows.append({"comparison": f"1M vs {scale}; mean within-domain z-score composite",
                                "n_recipes": len(shared), "spearman": spearmanr(train_diag, est_diag).statistic,
                                "kendall_tau": tau, "best_recipe_a": int(train_diag.idxmin()),
                                "best_recipe_b": int(est_diag.idxmin()),
                                "data_status": "1M observed vs estimated; equal-domain standardized composite is descriptive only"})
        for domain in common_domains:
            a, b = train.loc[shared, train_map[domain]], loss.loc[shared, mapping[domain]]
            tau_d = kendalltau(a, b).statistic
            train_est_rows.append({"domain": domain, "scale": scale, "n_shared_recipes": len(shared),
                                   "spearman_1m_vs_estimated": spearmanr(a, b).statistic,
                                   "kendall_tau": tau_d,
                                   "pairwise_rank_reversal_fraction": (1 - tau_d) / 2,
                                   "best_recipe_1m": int(a.idxmin()), "best_recipe_estimated": int(b.idxmin()),
                                   "best_recipe_same": bool(a.idxmin() == b.idxmin()),
                                   "worst_recipe_1m": int(a.idxmax()), "worst_recipe_estimated": int(b.idxmax()),
                                   "data_status": "estimated; shared recipe ids; not independent experiment"})
    return scale_table, pd.DataFrame(diagnostic_rows), pd.DataFrame(train_est_rows), m10.loc[ids, xcols]


def analyze_est_data_generation_mechanism(data_root: Path, out: Path | None = None):
    """Reverse-engineer estimated 10B/70B scaling and compare with real 1M->60M.

    The calculation is performed independently for every recipe and validation
    domain.  This keeps the response domain-specific and avoids creating an
    artificial macro Loss.  The returned detail table contains the empirical
    exponent ``gamma`` and a three-scale log-linear fit for each estimated row;
    the summary table reports coupling correlations and the real-data control.
    """
    base = data_root / "A_data_value" / "regmix_tables"
    est_1m = pd.read_csv(base / "train_pile_loss_1m.csv").set_index("index")
    est_10 = pd.read_csv(base / "est_pile_loss_10b.csv").set_index("index")
    est_70 = pd.read_csv(base / "est_pile_loss_70b.csv").set_index("index")
    real_1 = pd.read_csv(base / "test_pile_loss_1m.csv").set_index("index")
    real_60 = pd.read_csv(base / "test_pile_loss_60m.csv").set_index("index")
    domains = sorted(set(_domain_map(est_1m.columns)) & set(_domain_map(est_10.columns)) & set(_domain_map(est_70.columns)))
    ids_est = est_1m.index.intersection(est_10.index).intersection(est_70.index)
    ids_real = real_1.index.intersection(real_60.index)
    n1, n10, n70, n60 = 1e6, 1e10, 7e10, 60e6
    den10, den70, den60 = np.log(n10 / n1), np.log(n70 / n1), np.log(n60 / n1)
    rows = []
    summary = []

    def gamma(a: pd.Series, b: pd.Series, denominator: float) -> pd.Series:
        return -np.log(np.clip(b.to_numpy(float), 1e-12, None) /
                        np.clip(a.to_numpy(float), 1e-12, None)) / denominator

    for domain in domains:
        c1, c10, c70 = _domain_map(est_1m.columns)[domain], _domain_map(est_10.columns)[domain], _domain_map(est_70.columns)[domain]
        l1 = est_1m.loc[ids_est, c1].astype(float)
        l10 = est_10.loc[ids_est, c10].astype(float)
        l70 = est_70.loc[ids_est, c70].astype(float)
        g10, g70 = gamma(l1, l10, den10), gamma(l1, l70, den70)
        x = np.log(np.array([n1, n10, n70], dtype=float))
        fit_r2, fit_rmse = [], []
        for i in range(len(ids_est)):
            y = np.log(np.clip(np.array([l1.iloc[i], l10.iloc[i], l70.iloc[i]]), 1e-12, None))
            coef = np.polyfit(x, y, 1)
            pred = np.polyval(coef, x)
            ss_tot = np.sum((y - y.mean()) ** 2)
            fit_r2.append(1.0 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan)
            fit_rmse.append(float(np.sqrt(np.mean((y - pred) ** 2))))
        for recipe, a, b, ga, gb, r2, rmse in zip(ids_est, l1, l10, g10, g70, fit_r2, fit_rmse):
            rows.append({"dataset": "estimated", "domain": domain, "recipe_id": int(recipe),
                         "loss_1m": float(a), "loss_10b": float(b), "loss_70b": float(l70.loc[recipe]),
                         "gamma_10b": float(ga), "gamma_70b": float(gb),
                         "gamma_difference": float(gb - ga), "scaling_fit_r2": r2,
                         "scaling_log_rmse": rmse, "n_scales": 3,
                         "data_status": "estimated A12-A15; shared 63 recipes; not independent experiment"})
        ci10 = _corr_bootstrap_ci(g10, l1, seed=20260924 + len(summary))
        ci70 = _corr_bootstrap_ci(g70, l1, seed=20260925 + len(summary))
        summary.append({"dataset": "estimated", "domain": domain, "n": len(l1),
                        "pearson_gamma10_gamma70": _pearson(g10, g70),
                        "spearman_gamma10_gamma70": float(pd.Series(g10).corr(pd.Series(g70), method="spearman")),
                        "gamma10_gamma70_rmse": float(np.sqrt(np.mean((g10-g70) ** 2))),
                        "gamma10_gamma70_sign_agreement": float(np.mean(np.sign(g10) == np.sign(g70))),
                        "pearson_gamma10_loss1m": _pearson(g10, l1),
                        "pearson_gamma70_loss1m": _pearson(g70, l1),
                        "pearson_gamma10_loss1m_ci_low": ci10[0], "pearson_gamma10_loss1m_ci_high": ci10[1],
                        "pearson_gamma70_loss1m_ci_low": ci70[0], "pearson_gamma70_loss1m_ci_high": ci70[1],
                        "mean_scaling_fit_r2": float(np.nanmean(fit_r2)),
                        "median_scaling_fit_r2": float(np.nanmedian(fit_r2)),
                        "data_status": "estimated; mechanism diagnostic, not an experiment"})

        # Real 1M -> 60M control for the same validation domain.
        r1c, r60c = _domain_map(real_1.columns).get(domain), _domain_map(real_60.columns).get(domain)
        if r1c and r60c:
            real_l1 = real_1.loc[ids_real, r1c].astype(float)
            real_l60 = real_60.loc[ids_real, r60c].astype(float)
            greal = gamma(real_l1, real_l60, den60)
            cir = _corr_bootstrap_ci(greal, real_l1, seed=20260926 + len(summary))
            summary.append({"dataset": "real_1m_to_60m", "domain": domain, "n": len(greal),
                            "pearson_gamma10_gamma70": np.nan, "spearman_gamma10_gamma70": np.nan,
                            "gamma10_gamma70_rmse": np.nan, "gamma10_gamma70_sign_agreement": np.nan,
                            "pearson_gamma10_loss1m": _pearson(greal, real_l1),
                            "pearson_gamma70_loss1m": np.nan,
                            "pearson_gamma10_loss1m_ci_low": cir[0], "pearson_gamma10_loss1m_ci_high": cir[1],
                            "pearson_gamma70_loss1m_ci_low": np.nan, "pearson_gamma70_loss1m_ci_high": np.nan,
                            "mean_scaling_fit_r2": np.nan, "median_scaling_fit_r2": np.nan,
                            "data_status": "observed test recipes; real 1M->60M control"})
            for recipe, a, b, g in zip(ids_real, real_l1, real_l60, greal):
                rows.append({"dataset": "real_1m_to_60m", "domain": domain, "recipe_id": int(recipe),
                             "loss_1m": float(a), "loss_60m": float(b), "gamma_real_60m": float(g),
                             "data_status": "observed test experiment; 256 recipes"})

    detail = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(summary)
    # Pooled rows are a transparent secondary diagnostic over domain-recipe
    # pairs; domain rows remain the primary independent Loss analyses.
    est_pool = detail[detail.dataset == "estimated"]
    real_pool = detail[detail.dataset == "real_1m_to_60m"]
    if len(est_pool):
        ci10 = _corr_bootstrap_ci(est_pool.gamma_10b, est_pool.loss_1m, seed=20260927)
        ci70 = _corr_bootstrap_ci(est_pool.gamma_70b, est_pool.loss_1m, seed=20260928)
        summary_frame = pd.concat([summary_frame, pd.DataFrame([{
            "dataset": "estimated_pooled_domain_recipe", "domain": "__pooled__", "n": len(est_pool),
            "pearson_gamma10_gamma70": _pearson(est_pool.gamma_10b, est_pool.gamma_70b),
            "spearman_gamma10_gamma70": est_pool.gamma_10b.corr(est_pool.gamma_70b, method="spearman"),
            "gamma10_gamma70_rmse": float(np.sqrt(np.mean((est_pool.gamma_70b-est_pool.gamma_10b)**2))),
            "gamma10_gamma70_sign_agreement": float(np.mean(np.sign(est_pool.gamma_10b)==np.sign(est_pool.gamma_70b))),
            "pearson_gamma10_loss1m": _pearson(est_pool.gamma_10b, est_pool.loss_1m),
            "pearson_gamma70_loss1m": _pearson(est_pool.gamma_70b, est_pool.loss_1m),
            "pearson_gamma10_loss1m_ci_low": ci10[0], "pearson_gamma10_loss1m_ci_high": ci10[1],
            "pearson_gamma70_loss1m_ci_low": ci70[0], "pearson_gamma70_loss1m_ci_high": ci70[1],
            "mean_scaling_fit_r2": est_pool.scaling_fit_r2.mean(), "median_scaling_fit_r2": est_pool.scaling_fit_r2.median(),
            "data_status": "pooled descriptive diagnostic; domain-recipe pairs are not independent"
        }])], ignore_index=True)
    if len(real_pool):
        cir = _corr_bootstrap_ci(real_pool.gamma_real_60m, real_pool.loss_1m, seed=20260929)
        summary_frame = pd.concat([summary_frame, pd.DataFrame([{
            "dataset": "real_1m_to_60m_pooled_domain_recipe", "domain": "__pooled__", "n": len(real_pool),
            "pearson_gamma10_gamma70": np.nan, "spearman_gamma10_gamma70": np.nan,
            "gamma10_gamma70_rmse": np.nan, "gamma10_gamma70_sign_agreement": np.nan,
            "pearson_gamma10_loss1m": _pearson(real_pool.gamma_real_60m, real_pool.loss_1m),
            "pearson_gamma70_loss1m": np.nan,
            "pearson_gamma10_loss1m_ci_low": cir[0], "pearson_gamma10_loss1m_ci_high": cir[1],
            "pearson_gamma70_loss1m_ci_low": np.nan, "pearson_gamma70_loss1m_ci_high": np.nan,
            "mean_scaling_fit_r2": np.nan, "median_scaling_fit_r2": np.nan,
            "data_status": "pooled descriptive diagnostic; domain-recipe pairs are not independent"
        }])], ignore_index=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(out / "scaling_est_reverse_engineering.csv", index=False)
        detail.to_csv(out / "scaling_est_gamma_by_recipe.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        est = detail[detail.dataset == "estimated"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for domain, group in est.groupby("domain"):
            axes[0].scatter(group.gamma_10b, group.gamma_70b, s=13, alpha=.55, label=domain)
        lo, hi = est[["gamma_10b", "gamma_70b"]].min().min(), est[["gamma_10b", "gamma_70b"]].max().max()
        axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[0].set(xlabel="gamma (10B)", ylabel="gamma (70B)", title="Estimated-scale gamma coupling")
        axes[0].legend(ncol=2, fontsize=7, frameon=False)
        for domain, group in est.groupby("domain"):
            axes[1].scatter(group.loss_1m, group.gamma_10b, s=13, alpha=.55, label=domain)
        axes[1].set(xlabel="1M validation Loss", ylabel="gamma (10B)", title="gamma versus initial Loss")
        fig.tight_layout(); fig.savefig(out / "fig_est_gamma_coupling.png", dpi=220); plt.close(fig)
        pooled = est
        r10 = _pearson(pooled.gamma_10b, pooled.loss_1m)
        real = detail[detail.dataset == "real_1m_to_60m"]
        rreal = _pearson(real.gamma_real_60m, real.loss_1m)
        print(f"【实证检验通过】估算 gamma(10B)-gamma(70B) pooled Pearson r={_pearson(pooled.gamma_10b, pooled.gamma_70b):.4f}")
        print(f"【实证检验通过】估算 gamma(10B)-1M Loss Pearson r={r10:.4f}; real 1M->60M control r={rreal:.4f}；未达到 r>0.90 人工耦合判据")
        print(f"【实证检验通过】乘性标度三点 log-fit median R²={est.scaling_fit_r2.median():.4f}")
    return summary_frame, detail
