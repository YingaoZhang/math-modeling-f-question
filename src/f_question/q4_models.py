"""Statistical models for the efficiency frontier and technology decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def fit_stratified_frontier_calibration(task_scores: pd.DataFrame,
                                        leaderboard: pd.DataFrame,
                                        frontier: pd.DataFrame,
                                        n_boot: int = 1000,
                                        seed: int = 20260925) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calibrate a direct benchmark frontier with C8 task evidence.

    C8 is reduced to a per-model macro task score and joined to C1 by the
    normalized model name.  The direct C1 scale curve is the primary level;
    the C8 task index supplies a matched-model calibration and an uncertainty
    band.  This layered construction avoids treating seven High Loss bridge
    points as a universal benchmark ruler.
    """
    from .q4_task_aggregation import _key
    lb = leaderboard.copy()
    model_col = "Model" if "Model" in lb.columns else "model"
    avg_col = "Average" if "Average" in lb.columns else next(c for c in lb.columns if c.startswith("Average"))
    pcol = "N_params_B" if "N_params_B" in lb.columns else "#Params (B)"
    lb["_key"] = lb[model_col].map(_key)
    lb["benchmark"] = pd.to_numeric(lb[avg_col], errors="coerce")
    lb["params"] = pd.to_numeric(lb[pcol], errors="coerce")
    task = task_scores.copy()
    task["_key"] = task["model"].map(_key)
    ti = task.groupby("_key", as_index=False).agg(c8_index=("score", "mean"), n_tasks=("score", "count"))
    matched = lb.merge(ti, on="_key", how="inner").dropna(subset=["benchmark", "params", "c8_index"])
    matched = matched[(matched.params > 0) & (matched.n_tasks >= 3)]
    if len(matched) >= 5:
        X = np.column_stack([np.ones(len(matched)), matched.c8_index.to_numpy(float)])
        coef = np.linalg.lstsq(X, matched.benchmark.to_numpy(float), rcond=None)[0]
        matched["c8_calibrated"] = X @ coef
        resid = matched.benchmark - matched.c8_calibrated
        cal = {"n_matched": int(len(matched)), "intercept": float(coef[0]), "slope": float(coef[1]),
               "rmse": float(np.sqrt(np.mean(resid**2))), "benchmark_p05": float(lb.benchmark.quantile(.05)),
               "benchmark_p95": float(lb.benchmark.quantile(.95))}
    else:
        cal = {"n_matched": int(len(matched)), "intercept": float(lb.benchmark.mean()), "slope": 0.0,
               "rmse": float(lb.benchmark.std()), "benchmark_p05": float(lb.benchmark.quantile(.05)),
               "benchmark_p95": float(lb.benchmark.quantile(.95))}
        matched["c8_calibrated"] = cal["intercept"]
    # direct C1 scale relation, bounded to observed benchmark range
    valid = lb[(lb.params > 0) & lb.benchmark.notna()]
    xx = np.log10(valid.params.to_numpy(float)); yy = valid.benchmark.to_numpy(float)
    bx = np.column_stack([np.ones(len(xx)), xx])
    bcoef = np.linalg.lstsq(bx, yy, rcond=None)[0]
    cal.update({"direct_intercept": float(bcoef[0]), "direct_scale_slope": float(bcoef[1]),
                "calibration_method": "C1 direct scale + C8 matched task layer"})
    # A second, independent scale curve uses the matched C8 task index.  It
    # supplies the high-budget level where C1's public table is sparse.
    tx = matched["c8_index"].to_numpy(float) if len(matched) else np.array([])
    tp = matched["params"].to_numpy(float) if len(matched) else np.array([])
    if len(tx) >= 5:
        tcoef = np.linalg.lstsq(np.column_stack([np.ones(len(tx)), np.log10(tp)]), tx, rcond=None)[0]
        cal.update({"c8_scale_intercept": float(tcoef[0]), "c8_scale_slope": float(tcoef[1])})
    else:
        cal.update({"c8_scale_intercept": 0.0, "c8_scale_slope": 0.0})
    # forecast by the Q3 frontier's N coordinate and report a task-adjusted
    # score using the C8 calibrated layer as a cross-check.
    fr = frontier.copy()
    fr["N_params_B"] = pd.to_numeric(fr["N_params_B"], errors="coerce")
    fr["direct_c1_score"] = np.clip(bcoef[0] + bcoef[1]*np.log10(fr["N_params_B"].clip(lower=1e-6)), cal["benchmark_p05"], cal["benchmark_p95"])
    task_scale = cal["c8_scale_intercept"] + cal["c8_scale_slope"] * np.log10(fr["N_params_B"].clip(lower=1e-6))
    fr["direct_score_raw"] = cal["intercept"] + cal["slope"] * task_scale
    fr["direct_score"] = np.clip(fr["direct_score_raw"], 40.0, 70.0)
    fr["c8_layer_score"] = np.clip(cal["intercept"] + cal["slope"] * (cal["intercept"]*0 + matched.c8_index.median() if len(matched) else 0), cal["benchmark_p05"], cal["benchmark_p95"])
    # Bootstrap uncertainty of the matched C8 calibration.
    rng = np.random.default_rng(seed); draws = []
    if len(matched) >= 5:
        for i in range(n_boot):
            ids = rng.integers(0, len(matched), len(matched))
            Xb = np.column_stack([np.ones(len(ids)), matched.c8_index.to_numpy(float)[ids]])
            cb = np.linalg.lstsq(Xb, matched.benchmark.to_numpy(float)[ids], rcond=None)[0]
            draws.append({"draw": i, "intercept": cb[0], "slope": cb[1]})
    boot = pd.DataFrame(draws)
    return fr[[c for c in ["budget_flops", "N_params_B", "direct_score_raw", "direct_score", "c8_layer_score"] if c in fr]], boot, cal


def _bounded_logistic(x: np.ndarray, floor: float, ceiling: float,
                      slope: float, midpoint: float) -> np.ndarray:
    """Bounded S-shaped loss frontier on log10 compute."""
    z = np.clip(slope * (x - midpoint), -60.0, 60.0)
    return floor + (ceiling - floor) / (1.0 + np.exp(z))


def fit_bounded_frontier(frontier: pd.DataFrame) -> tuple[dict, float, str]:
    """Fit a bounded logistic frontier and return parameters, RMSE and status.

    The fit is deliberately a sensitivity model: it prevents endpoint clamping
    from creating a constant future score while retaining a transparent
    fallback to log-linear interpolation when the short curve cannot identify
    all four parameters.
    """
    f = frontier.sort_values("budget_flops").drop_duplicates("budget_flops")
    x = np.log10(pd.to_numeric(f.budget_flops, errors="coerce").to_numpy(float))
    y = pd.to_numeric(f.loss, errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 6 or np.ptp(y) < 1e-8:
        return {}, float("nan"), "insufficient_variation_fallback"
    lo, hi = float(np.min(y) - .5), float(np.max(y) + .5)
    p0 = [max(0.0, float(np.min(y))), float(np.max(y)), 1.0, float(np.median(x))]
    try:
        pars, _ = curve_fit(_bounded_logistic, x, y, p0=p0,
                            bounds=([0.0, 0.0, 1e-4, float(np.min(x)-3)],
                                    [float(np.max(y)+1), float(np.max(y)+2), 20.0, float(np.max(x)+3)]),
                            maxfev=20000)
        pred = _bounded_logistic(x, *pars)
        rmse = float(np.sqrt(np.mean((y-pred)**2)))
        return {"floor": float(pars[0]), "ceiling": float(pars[1]),
                "slope": float(pars[2]), "midpoint_log10_flops": float(pars[3])}, rmse, "bounded_logistic"
    except Exception:
        return {}, float("nan"), "fit_failed_fallback"


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
                      recent_compute: float, horizons=(12, 24), rates=(1.0, 1.2, 2.0, 4.0),
                      bridge_boot: pd.DataFrame | None = None,
                      loss_bias_mean: float = 0.0,
                      loss_bias_ci: tuple[float, float] | None = None,
                      rho: float = 0.0) -> pd.DataFrame:
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
    frontier_params, frontier_rmse, frontier_model = fit_bounded_frontier(f)
    # Treat values within floating-point roundoff of the Q3 support boundary
    # as boundary points, not as out-of-support extrapolations.
    support_tol = 1e-10
    loss_at_min, loss_at_max = float(y[0]), float(y[-1])
    rows = []
    for h in horizons:
        for annual in rates:
            # Effective compute includes time-dependent progress estimated as
            # C_eff(t)=C(t) exp(rho*t); rho is calibrated from the C1
            # scale/time regression and is set to zero when the estimate is
            # unavailable. This separates hardware growth from algorithmic
            # progress in the scenario table.
            months = float(h)
            c_nominal = recent_compute * annual ** (months/12)
            c = c_nominal * np.exp(rho * months)
            clipped_c = float(np.clip(c, support_min, support_max))
            below = c < support_min * (1.0 - support_tol)
            above = c > support_max * (1.0 + support_tol)
            extrap = bool(below or above)
            if frontier_model == "bounded_logistic":
                loss = float(_bounded_logistic(np.array([np.log10(c)]), frontier_params["floor"],
                                               frontier_params["ceiling"], frontier_params["slope"],
                                               frontier_params["midpoint_log10_flops"])[0])
            else:
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
            rows.append({"horizon_months": h, "annual_compute_growth": annual,
                         "nominal_compute_flops": c_nominal, "effective_compute_flops": c,
                         "compute_flops": c,
                         "tech_progress_rho_per_month": rho,
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
                         "frontier_model": frontier_model,
                         "frontier_fit_rmse": frontier_rmse,
                         "frontier_floor": frontier_params.get("floor", np.nan),
                         "frontier_ceiling": frontier_params.get("ceiling", np.nan),
                         "outside_high_bridge_loss_support": not (
                             bridge_summary["high_loss_min"] <= loss <= bridge_summary["high_loss_max"]
                         ), "bridge_n_high": bridge_summary["n_high"]})
    return pd.DataFrame(rows)
