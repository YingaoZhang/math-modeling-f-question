"""Statistical models for the efficiency frontier and technology decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fit_loss_benchmark_bridge(bridge: pd.DataFrame, n_boot: int = 2000,
                              seed: int = 20260925) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit an affine Pythia-loss to benchmark bridge on High rows only.

    Medium rows are retained for direction/ranking checks and never enter the
    calibration.  With only seven High observations, bootstrap intervals are
    intentionally reported as wide uncertainty rather than hidden.
    """
    d = bridge[bridge["Loss_Comparability"].astype(str).str.startswith("High")].copy()
    d["Val_Loss"] = pd.to_numeric(d.Val_Loss, errors="coerce")
    d["LB_Average"] = pd.to_numeric(d.LB_Average, errors="coerce")
    d = d.dropna(subset=["Val_Loss", "LB_Average"]).reset_index(drop=True)
    x, y = d.Val_Loss.to_numpy(float), d.LB_Average.to_numpy(float)
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    pred = intercept + slope * x
    residuals = d.assign(predicted=pred, residual=y-pred)
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        if len(np.unique(idx)) < 2:
            continue
        ss, ii, *_ = stats.linregress(x[idx], y[idx])
        draws.append({"draw": i, "slope": ss, "intercept": ii})
    boot = pd.DataFrame(draws)
    ci = {"slope_low": float(boot.slope.quantile(.025)), "slope_high": float(boot.slope.quantile(.975)),
          "intercept_low": float(boot.intercept.quantile(.025)), "intercept_high": float(boot.intercept.quantile(.975))}
    summary = {"n_high": len(d), "n_all": len(bridge), "slope": float(slope), "intercept": float(intercept),
               "r2": float(r*r), "f_pvalue": float(p), "residual_rmse": float(np.sqrt(np.mean((y-pred)**2))), **ci}
    summary["high_loss_min"] = float(x.min())
    summary["high_loss_max"] = float(x.max())
    return residuals, boot, summary


def decompose_scale_vs_time(leaderboard: pd.DataFrame, n_boot: int = 2000,
                            seed: int = 20260925) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Regress benchmark on scale and time and bootstrap contribution shares.

    Variance shares use the two-predictor Shapley decomposition of R-squared;
    residual share is 1 - full-model R-squared. SD ratios are also retained
    as a distinct descriptive convention, not called variance attribution.
    """
    d = leaderboard.copy()
    d["months"] = (d.date - d.date.min()).dt.days / 30.4375
    d = d.dropna(subset=["months", "params_b", "average"])
    scale = np.log10(d.params_b.to_numpy(float))
    time = d.months.to_numpy(float)
    y = d.average.to_numpy(float)
    X = np.column_stack([np.ones(len(d)), scale, time])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    d["predicted"] = pred
    d["scale_contribution"] = beta[1] * (np.log10(d.params_b) - np.log10(d.params_b).mean())
    d["time_contribution"] = beta[2] * (d.months - d.months.mean())
    ss_res = float(np.sum((y-pred)**2)); ss_tot = float(np.sum((y-y.mean())**2))
    def r2_for(features: list[np.ndarray]) -> float:
        xx = np.column_stack([np.ones(len(y)), *features])
        bb, *_ = np.linalg.lstsq(xx, y, rcond=None)
        rr = y-xx@bb
        return 1-float(rr@rr)/ss_tot if ss_tot else np.nan

    r2_scale, r2_time, r2_full = r2_for([scale]), r2_for([time]), 1-ss_res/ss_tot if ss_tot else np.nan
    shapley_scale = .5*(r2_scale + r2_full-r2_time)
    shapley_time = .5*(r2_time + r2_full-r2_scale)
    summary = {"n_pretrained": len(d), "intercept": float(beta[0]), "scale_coef": float(beta[1]),
               "time_coef_per_month": float(beta[2]), "r2": 1-ss_res/ss_tot if ss_tot else np.nan,
               "scale_sd_contribution": float(d.scale_contribution.std()),
               "time_sd_contribution": float(d.time_contribution.std()),
               "r2_scale_only": r2_scale, "r2_time_only": r2_time,
               "variance_share_scale": shapley_scale,
               "variance_share_time": shapley_time,
               "variance_share_residual": 1-r2_full,
               "sd_ratio_scale": float(d.scale_contribution.std() / (d.scale_contribution.std()+d.time_contribution.std())),
               "sd_ratio_time": float(d.time_contribution.std() / (d.scale_contribution.std()+d.time_contribution.std()))}

    rng = np.random.default_rng(seed)
    draws = []
    for draw in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        ys, xs, ts = y[idx], scale[idx], time[idx]
        xx = np.column_stack([np.ones(len(idx)), xs, ts])
        bb, *_ = np.linalg.lstsq(xx, ys, rcond=None)
        total = float(np.sum((ys-ys.mean())**2))
        if total <= 0:
            continue
        def boot_r2(features: list[np.ndarray]) -> float:
            mat = np.column_stack([np.ones(len(idx)), *features])
            coef, *_ = np.linalg.lstsq(mat, ys, rcond=None)
            return 1-float(np.sum((ys-mat@coef)**2))/total
        rs, rt, rf = boot_r2([xs]), boot_r2([ts]), boot_r2([xs, ts])
        sc = .5*(rs+rf-rt); tc = .5*(rt+rf-rs)
        draws.append({"draw": draw, "scale_coef": bb[1], "time_coef_per_month": bb[2],
                      "variance_share_scale": sc, "variance_share_time": tc,
                      "variance_share_residual": 1-rf})
    boot = pd.DataFrame(draws)
    for col in ["scale_coef", "time_coef_per_month", "variance_share_scale", "variance_share_time", "variance_share_residual"]:
        summary[f"{col}_ci_low"] = float(boot[col].quantile(.025))
        summary[f"{col}_ci_high"] = float(boot[col].quantile(.975))
    return d, summary, boot


def forecast_frontier(frontier: pd.DataFrame, bridge_summary: dict,
                      recent_compute: float, horizons=(12, 24), rates=(1.25, 2.0, 4.0),
                      bridge_boot: pd.DataFrame | None = None,
                      loss_bias_mean: float = 0.0,
                      loss_bias_ci: tuple[float, float] | None = None) -> pd.DataFrame:
    """Translate Q3 Loss*(C) to benchmark scenarios with uncertainty labels.

    Bridge intervals come from resampling the small High-comparability set.
    B9/B10 residual bias is reported as a separate same-table sensitivity and
    is never silently applied to the central prediction.
    """
    f = frontier.sort_values("budget_flops").drop_duplicates("budget_flops")
    budget = pd.to_numeric(f.budget_flops, errors="coerce").to_numpy(float)
    x = np.log10(budget)
    y = pd.to_numeric(f.loss, errors="coerce").to_numpy(float)
    support_min, support_max = float(budget.min()), float(budget.max())
    # Treat values within floating-point roundoff of the Q3 support boundary
    # as boundary points, not as out-of-support extrapolations.
    support_tol = 1e-10
    loss_at_min, loss_at_max = float(y[0]), float(y[-1])
    rows = []
    for h in horizons:
        for annual in rates:
            c = recent_compute * annual ** (h/12)
            clipped_c = float(np.clip(c, support_min, support_max))
            below = c < support_min * (1.0 - support_tol)
            above = c > support_max * (1.0 + support_tol)
            extrap = bool(below or above)
            loss = float(np.interp(np.log10(clipped_c), x, y))
            if below:
                endpoint = "lower_endpoint"
                endpoint_multiplier = support_min / c
            elif above:
                endpoint = "upper_endpoint"
                endpoint_multiplier = c / support_max
            elif np.isclose(c, support_min, rtol=support_tol, atol=0.0):
                endpoint = "lower_endpoint"
                endpoint_multiplier = 1.0
            elif np.isclose(c, support_max, rtol=support_tol, atol=0.0):
                endpoint = "upper_endpoint"
                endpoint_multiplier = 1.0
            else:
                endpoint = "interpolated"
                endpoint_multiplier = 1.0
            score = bridge_summary["intercept"] + bridge_summary["slope"] * loss
            if bridge_boot is not None and not bridge_boot.empty:
                bridge_scores = (bridge_boot["intercept"].to_numpy(float)
                                 + bridge_boot["slope"].to_numpy(float) * loss)
                score_ci_low, score_ci_high = np.quantile(bridge_scores, [.025, .975])
            else:
                score_ci_low = score_ci_high = score
            bias_ci = loss_bias_ci or (loss_bias_mean, loss_bias_mean)
            biased_losses = loss + np.array([bias_ci[0], bias_ci[1]], dtype=float)
            bias_scores = bridge_summary["intercept"] + bridge_summary["slope"] * biased_losses
            rows.append({"horizon_months": h, "annual_compute_growth": annual, "compute_flops": c,
                         "loss_star": loss, "benchmark_predicted": score, "outside_q3_support": extrap,
                         "q3_support_min_flops": support_min, "q3_support_max_flops": support_max,
                         "q3_clipped_compute_flops": clipped_c,
                         "q3_endpoint": endpoint,
                         "q3_endpoint_multiplier": float(endpoint_multiplier),
                         "q3_endpoint_loss": loss_at_min if endpoint == "lower_endpoint" else (loss_at_max if endpoint == "upper_endpoint" else np.nan),
                         "bridge_bootstrap_95ci_low": float(score_ci_low),
                         "bridge_bootstrap_95ci_high": float(score_ci_high),
                         "b9_b10_bias_mean_nats_same_table": float(loss_bias_mean),
                         "bias_sensitivity_loss_low": float(np.min(biased_losses)),
                         "bias_sensitivity_loss_high": float(np.max(biased_losses)),
                         "bias_sensitivity_benchmark_low": float(np.min(bias_scores)),
                         "bias_sensitivity_benchmark_high": float(np.max(bias_scores)),
                         "outside_high_bridge_loss_support": not (
                             bridge_summary["high_loss_min"] <= loss <= bridge_summary["high_loss_max"]
                         ), "bridge_n_high": bridge_summary["n_high"]})
    return pd.DataFrame(rows)
